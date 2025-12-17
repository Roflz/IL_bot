# trade.py (actions)

from __future__ import annotations
from typing import List, Dict

from .widgets import get_widget_children, click_widget
from helpers.runtime_utils import dispatch
from helpers.utils import clean_rs, rect_beta_xy


def get_other_offer() -> List[Dict]:
    """
    Get the other player's trade offer items.
    
    Returns:
        A list of dictionaries containing the other player's offer items.
    """
    try:
        children_widgets = get_widget_children(21954588)
        
        if not children_widgets:
            return []
        
        # Filter children that have a 'text' field
        offer_items = [
            widget for widget in children_widgets.get("children", [])
            if widget.get("name")
        ]
        
        return offer_items
        
    except Exception as e:
        print(f"[TRADE] Error getting other offer: {e}")
        return []


def other_offer_contains(item_name: str) -> bool:
    """
    Check if a certain item exists in the other player's offer.
    
    Args:
        item_name: The name of the item to search for
        
    Returns:
        True if the item is found in the other offer, False otherwise
    """
    try:
        offer_items = get_other_offer()
        
        for item in offer_items:
            if item_name in item.get("name").lower():
                return True
        
        return False
        
    except Exception as e:
        print(f"[TRADE] Error checking if other offer contains '{item_name}': {e}")
        return False


def get_my_offer() -> List[Dict]:
    """
    Get my trade offer items.
    
    Returns:
        A list of dictionaries containing my offer items.
    """
    try:
        children_widgets = get_widget_children(21954585)
        
        if not children_widgets:
            return []
        
        # Filter children that have a 'name' field
        offer_items = [
            widget for widget in children_widgets.get("children", [])
            if widget.get("name")
        ]
        
        return offer_items
        
    except Exception as e:
        print(f"[TRADE] Error getting my offer: {e}")
        return []


def my_offer_contains(item_name: str) -> bool:
    """
    Check if a certain item exists in my trade offer.
    
    Args:
        item_name: The name of the item to search for
        
    Returns:
        True if the item is found in my offer, False otherwise
    """
    try:
        offer_items = get_my_offer()
        
        for item in offer_items:
            if item_name == clean_rs(item.get("name")).lower():
                return True
        
        return False
        
    except Exception as e:
        print(f"[TRADE] Error checking if my offer contains '{item_name}': {e}")
        return False

def accept_trade():
    return click_widget(21954570)

def accept_trade_confirm():
    return click_widget(21889037)


def get_other_offer_confirmation() -> List[Dict]:
    """
    Get the other player's trade offer items on the confirmation screen.
    
    Returns:
        A list of dictionaries containing the other player's offer items.
    """
    try:
        children_widgets = get_widget_children(21889053)
        
        if not children_widgets:
            return []
        
        # Filter children that have a 'name' field
        offer_items = [
            widget for widget in children_widgets.get("children", [])
            if widget.get("text")
        ]
        
        return offer_items
        
    except Exception as e:
        print(f"[TRADE] Error getting other offer confirmation: {e}")
        return []


def other_offer_confirmation_contains(item_name: str) -> bool:
    """
    Check if a certain item exists in the other player's offer on the confirmation screen.
    
    Args:
        item_name: The name of the item to search for
        
    Returns:
        True if the item is found in the other offer, False otherwise
    """
    try:
        offer_items = get_other_offer_confirmation()
        
        for item in offer_items:
            if item_name in item.get("text").lower():
                return True
        
        return False
        
    except Exception as e:
        print(f"[TRADE] Error checking if other offer confirmation contains '{item_name}': {e}")
        return False


def get_my_offer_confirmation() -> List[Dict]:
    """
    Get my trade offer items on the confirmation screen.
    
    Returns:
        A list of dictionaries containing my offer items.
    """
    try:
        children_widgets = get_widget_children(21889052)
        
        if not children_widgets:
            return []
        
        # Filter children that have a 'name' field
        offer_items = [
            widget for widget in children_widgets.get("children", [])
            if widget.get("text")
        ]
        
        return offer_items
        
    except Exception as e:
        print(f"[TRADE] Error getting my offer confirmation: {e}")
        return []


def my_offer_confirmation_contains(item_name: str) -> bool:
    """
    Check if a certain item exists in my trade offer on the confirmation screen.
    
    Args:
        item_name: The name of the item to search for
        
    Returns:
        True if the item is found in my offer, False otherwise
    """
    try:
        offer_items = get_my_offer_confirmation()
        
        for item in offer_items:
            if item_name in item.get("text").lower():
                return True
        
        return False
        
    except Exception as e:
        print(f"[TRADE] Error checking if my offer confirmation contains '{item_name}': {e}")
        return False


def get_trade_inventory() -> List[Dict]:
    """
    Get the trade inventory items.
    
    Returns:
        A list of dictionaries containing the trade inventory items.
    """
    try:
        children_widgets = get_widget_children(22020096)
        
        if not children_widgets:
            return []
        
        # Filter children that have a 'name' field
        inventory_items = [
            widget for widget in children_widgets.get("children", [])
            if widget.get("name")
        ]
        
        return inventory_items
        
    except Exception as e:
        print(f"[TRADE] Error getting trade inventory: {e}")
        return []


def offer_item(item_name: str) -> bool:
    """
    Offer 1 of an item from the trade inventory.
    
    Args:
        item_name: The name of the item to offer
        
    Returns:
        True if the offer was successful, False otherwise
    """
    return _offer_item_quantity(item_name, "offer", 1)


def offer_5_items(item_name: str) -> bool:
    """
    Offer 5 of an item from the trade inventory.
    
    Args:
        item_name: The name of the item to offer
        
    Returns:
        True if the offer was successful, False otherwise
    """
    return _offer_item_quantity(item_name, "offer-5", 5)


def offer_10_items(item_name: str) -> bool:
    """
    Offer 10 of an item from the trade inventory.
    
    Args:
        item_name: The name of the item to offer
        
    Returns:
        True if the offer was successful, False otherwise
    """
    return _offer_item_quantity(item_name, "offer-10", 10)


def offer_x_items(item_name: str, quantity: int) -> bool:
    """
    Offer X of an item from the trade inventory.
    
    Args:
        item_name: The name of the item to offer
        quantity: The quantity to offer
        
    Returns:
        True if the offer was successful, False otherwise
    """
    return _offer_item_quantity(item_name, "offer-x", quantity)


def offer_all_items(item_name: str) -> bool:
    """
    Offer all of an item from the trade inventory.
    
    Args:
        item_name: The name of the item to offer
        
    Returns:
        True if the offer was successful, False otherwise
    """
    return _offer_item_quantity(item_name, "offer-all", -1)


def _offer_item_quantity(item_name: str, option: str, quantity: int) -> bool:
    """
    Internal method to offer a specific quantity of an item.
    
    Args:
        item_name: The name of the item to offer
        option: The menu option to select (offer, offer-5, etc.)
        quantity: The quantity being offered (for logging)
        
    Returns:
        True if the offer was successful, False otherwise
    """
    try:
        # Get trade inventory items
        inventory_items = get_trade_inventory()
        if not inventory_items:
            print(f"[TRADE] No trade inventory items found")
            return False
        
        # Find the item in the inventory
        target_item = None
        for item in inventory_items:
            if item_name.lower() in item.get("name", "").lower():
                target_item = item
                break
        
        if not target_item:
            print(f"[TRADE] Item '{item_name}' not found in trade inventory")
            return False
        
        # Get item coordinates
        bounds = target_item.get("bounds", {})
        if not bounds:
            print(f"[TRADE] No bounds found for item '{item_name}'")
            return False
        
        # Calculate center coordinates
        x, y = rect_beta_xy((bounds.get("x", 0), bounds.get("x", 0) + bounds.get("width", 0),
                             bounds.get("y", 0), bounds.get("y", 0) + bounds.get("height", 0)), alpha=2.0, beta=2.0)
        
        print(f"[TRADE] Offering {quantity} {item_name} at ({x}, {y})")
        
        # Create step for context clicking the item
        step = {
            "action": "click-item-context",
            "option": option,
            "click": {
                "type": "context-select",
                "x": x,
                "y": y,
                "row_height": 16,
                "start_dy": 10,
                "open_delay_ms": 120,
                "exact_match": True,
            },
            "target": {"domain": "item", "name": item_name},
            "anchor": {"x": x, "y": y}
        }
        
        result = dispatch(step)
        
        if result and result.get("click_result"):
            print(f"[TRADE] Successfully offered {quantity} {item_name}")
            return True
        else:
            print(f"[TRADE] Failed to offer {quantity} {item_name}")
            return False
        
    except Exception as e:
        print(f"[TRADE] Error offering {quantity} {item_name}: {e}")
        return False


def get_players() -> List[Dict]:
    """
    Get information about all players around the local player.
    
    Returns:
        A list of dictionaries containing player information including location, bounds, etc.
    """
    try:
        from ..helpers.runtime_utils import ipc
        
        result = ipc.get_players()
        
        if not result or not result.get("ok"):
            print(f"[PLAYERS] Failed to get players: {result}")
            return []
        
        return result.get("players", [])
        
    except Exception as e:
        print(f"[PLAYERS] Error getting players: {e}")
        return []


def find_player_by_name(search_name: str) -> Dict:
    """
    Find a player by name in the nearby players.
    
    Args:
        search_name: The name of the player to search for
        
    Returns:
        The player dictionary if found, None otherwise
    """
    try:
        players = get_players()
        if not players:
            print(f"[PLAYERS] No players found around character")
            return None

        for player in players:
            name = player.get("name")
            if name == search_name:
                return player
        
        return None
        
    except Exception as e:
        print(f"[PLAYERS] Error finding player '{search_name}': {e}")
        return None


def trade_with_player(search_name: str) -> bool:
    """
    Find a player by name and initiate a trade with them.
    
    Args:
        search_name: The name of the player to trade with
        
    Returns:
        True if trade was initiated successfully, False otherwise
    """
    try:
        # Find the player
        player = find_player_by_name(search_name)
        if not player:
            print(f"[TRADE] Player '{search_name}' not found nearby")
            return False
        
        player_name = player.get("name", "Unknown")
        canvas_x = player.get("canvasX")
        canvas_y = player.get("canvasY")
        
        if canvas_x is None or canvas_y is None:
            print(f"[TRADE] Player {player_name} has no canvas coordinates")
            return False
        
        print(f"[TRADE] Attempting to trade with player: {player_name} at ({canvas_x}, {canvas_y})")
        
        # Create a step for context clicking the player
        step = {
            "action": "click-player-context",
            "option": "Trade with",
            "click": {
                "type": "context-select",
                "x": canvas_x,
                "y": canvas_y,
                "row_height": 16,
                "start_dy": 10,
                "open_delay_ms": 120,
                "exact_match": False,
            },
            "target": {"domain": "player", "name": player_name},
            "anchor": {"x": canvas_x, "y": canvas_y}
        }

        result = dispatch(step)
        
        if result and result.get("click_result"):
            print(f"[TRADE] Successfully initiated trade with {player_name}")
            return True
        else:
            print(f"[TRADE] Failed to initiate trade with {player_name}")
            return False
        
    except Exception as e:
        print(f"[TRADE] Error trading with player '{search_name}': {e}")
        return False

