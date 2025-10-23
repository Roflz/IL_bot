#!/usr/bin/env python3
"""
Grand Exchange Utility Plan
==========================

This is a utility plan that can be used by other plans to buy and/or sell items from the Grand Exchange.
It takes lists of items to buy and sell with their configurations, then handles the entire GE process.

The plan follows this flow:
1. TRAVEL_TO_GE - Travel to Grand Exchange
2. SELL_ITEMS - Sell all required items (if any)
3. CHECK_COINS - Check available coins for buying (if buying items)
4. BUY_ITEMS - Buy all required items (if any)
5. CLOSE_GE - Close the Grand Exchange interface
6. COMPLETE - All done, return SUCCESS

Usage:
    # Example 1: Buy and sell items
    ge_plan = GePlan(
        items_to_buy=[
            {"name": "Trout", "quantity": 50, "bumps": 5, "set_price": 0},
            {"name": "Bronze scimitar", "quantity": 1, "bumps": 0, "set_price": 1000}
        ],
        items_to_sell=[
            {"name": "Raw beef", "quantity": 100, "bumps": 0, "set_price": 0}
        ]
    )
    
    # Example 2: Sell only
    sell_plan = GePlan(items_to_sell=[
        {"name": "Raw beef", "quantity": 100, "bumps": 0, "set_price": 0},
        {"name": "Cowhide", "quantity": 50, "bumps": 0, "set_price": 0}
    ])
    
    # Example 3: Sell all of specific items
    sell_all_plan = GePlan(items_to_sell=[
        {"name": "Raw beef", "quantity": -1, "bumps": 0, "set_price": 0},  # Sell ALL raw beef
        {"name": "Cowhide", "quantity": -1, "bumps": 0, "set_price": 0}   # Sell ALL cowhide
    ])
    
    # Example 3: Buy only (existing functionality)
    buy_plan = GePlan(items_to_buy=[
        {"name": "Trout", "quantity": 50, "bumps": 5, "set_price": 0}
    ])
    
    # In your plan loop:
    status = ge_plan.loop(ui)
    if status == GePlan.SUCCESS:
        print("All items sold/purchased and GE closed!")
    elif status == GePlan.SELLING:
        print("Still selling items...")
    elif status == GePlan.BUYING:
        print("Still buying items...")
    elif status == GePlan.ERROR:
        print("Error with GE operations")
"""

import time
import logging
from pathlib import Path
from typing import List, Dict, Any

# Add the parent directory to the path for imports
import sys

from ...actions.timing import wait_until

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ..base import Plan
from ...actions import ge, travel, player
from ...actions.ge import check_and_buy_required_items, check_and_sell_required_items, close_ge
from ...actions.travel import go_to, in_area


class GePlan(Plan):
    """Grand Exchange utility plan for buying items."""
    
    id = "GE_PLAN"
    label = "Grand Exchange Utility"
    
    # Return status codes
    SUCCESS = 0
    TRAVELING = 1
    CHECKING_COINS = 2
    SELLING = 3
    BUYING = 4
    ERROR = 5
    WAITING = 6
    INSUFFICIENT_FUNDS = 7
    
    def __init__(self, items_to_buy: List[Dict[str, Any]] = None, items_to_sell: List[Dict[str, Any]] = None):
        """
        Initialize the GE plan.
        
        Args:
            items_to_buy: List of items to buy, each with:
                - name: Item name
                - quantity: How many to buy
                - bumps: Number of price bumps (0 = use set_price)
                - set_price: Fixed price (0 = use bumps)
            items_to_sell: List of items to sell, each with:
                - name: Item name
                - quantity: How many to sell (use -1 for "sell all")
                - bumps: Number of price bumps (0 = use set_price)
                - set_price: Fixed price (0 = use bumps)
        """
        self.state = {"phase": "TRAVEL_TO_GE"}
        self.next = self.state["phase"]
        self.loop_interval_ms = 600
        
        # Set up camera immediately during initialization
        try:
            from ...helpers.camera import setup_camera_optimal
            setup_camera_optimal()
        except Exception as e:
            logging.warning(f"[{self.id}] Could not setup camera: {e}")
        
        # Items to buy configuration
        self.items_to_buy = items_to_buy or []
        
        # Items to sell configuration
        self.items_to_sell = items_to_sell or []
        
        # Convert items to the format expected by check_and_buy_required_items
        self.required_items = []
        self.item_requirements = {}
        
        for item_config in self.items_to_buy:
            item_name = item_config["name"]
            self.required_items.append(item_name)
            self.item_requirements[item_name] = (
                item_config["quantity"],
                item_config["bumps"],
                item_config["set_price"]
            )
        
        # Convert sell items to the format expected by sell functions
        self.sell_items = []
        self.sell_requirements = {}
        
        for item_config in self.items_to_sell:
            item_name = item_config["name"]
            quantity = item_config["quantity"]
            
            self.sell_items.append(item_name)
            self.sell_requirements[item_name] = (
                quantity,
                item_config["bumps"],
                item_config["set_price"]
            )
        
        # State tracking
        self.buying_complete = False
        self.selling_complete = False
        self.error_message = None
        self.coins_checked = False
        self.fallback_items = {}  # Items we can fall back to if we can't afford the best
        
        logging.info(f"[{self.id}] GE plan initialized")
        logging.info(f"[{self.id}] Items to buy: {[item['name'] for item in self.items_to_buy]}")
        logging.info(f"[{self.id}] Items to sell: {[item['name'] for item in self.items_to_sell]}")
    
    def set_phase(self, phase: str, camera_setup: bool = True):
        """Set the current phase."""
        from ...helpers.phase_utils import set_phase_with_camera
        return set_phase_with_camera(self, phase, camera_setup)
    
    def loop(self, ui) -> int:
        """Main loop method."""
        logged_in = player.logged_in()
        if not logged_in:
            player.login()
            return self.loop_interval_ms

        if self.buying_complete and self.selling_complete:
            return self.SUCCESS
        
        phase = self.state.get("phase", "TRAVEL_TO_GE")
        logging.debug(f"[{self.id}] Current phase: {phase}")

        match(phase):
            case "TRAVEL_TO_GE":
                return self._handle_travel_to_ge()

            case "CHECK_COINS":
                return self._handle_check_coins()

            case "SELL_ITEMS":
                return self._handle_sell_items()

            case "BUY_ITEMS":
                return self._handle_buy_items()

            case "CLOSE_GE":
                return self._handle_close_ge()

            case "COMPLETE":
                logging.info(f"[{self.id}] All items sold and purchased successfully!")
                self.buying_complete = True
                self.selling_complete = True
                return self.SUCCESS

        logging.warning(f"[{self.id}] Unknown phase: {phase}")
        return self.loop_interval_ms
    
    def _handle_travel_to_ge(self) -> int:
        """Handle traveling to the Grand Exchange."""
        if not in_area("GE"):
            logging.info(f"[{self.id}] Traveling to Grand Exchange...")
            go_to("GE")
            return self.TRAVELING
        else:
            logging.info(f"[{self.id}] Arrived at Grand Exchange")
            # If we have items to sell, sell first, otherwise check coins for buying
            if self.items_to_sell:
                self.set_phase("SELL_ITEMS")
            else:
                self.set_phase("CHECK_COINS")
            return self.loop_interval_ms
    
    def _handle_check_coins(self) -> int:
        """Handle checking if we have enough coins for purchases."""
        logging.info(f"[{self.id}] Checking available coins...")
        
        try:
            from ...actions import bank
            
            # Get available coins from bank
            if not bank.is_open():
                bank.open_bank()
                time.sleep(1)
                return self.CHECKING_COINS

            # Deposit any coins from inventory to bank first
            from ...actions import inventory
            if inventory.has_item("Coins"):
                logging.info(f"[{self.id}] Depositing coins from inventory to bank")
                bank.deposit_inventory()
                if not wait_until(lambda: not inventory.has_item("Coins")):
                    return None
                time.sleep(1)
            
            # Get coin count from bank
            coin_count = 0
            try:
                coin_count = bank.get_item_count("Coins")
                logging.info(f"[{self.id}] Found {coin_count} coins in bank")
            except Exception as e:
                logging.warning(f"[{self.id}] Could not get coin count from bank: {e}")
                coin_count = 0
            
            logging.info(f"[{self.id}] Available coins: {coin_count}")
            
            # Calculate total cost of items we want to buy
            total_cost = 0
            affordable_items = []
            unaffordable_items = []
            
            for item_config in self.items_to_buy:
                item_name = item_config["name"]
                quantity = item_config["quantity"]
                set_price = item_config["set_price"]
                
                # Estimate cost (rough approximation)
                if set_price > 0:
                    item_cost = set_price * quantity
                else:
                    item_cost = 0
                
                total_cost += item_cost
                
                if coin_count >= item_cost:
                    affordable_items.append(item_config)
                    logging.info(f"[{self.id}] Can afford {item_name} (cost: ~{item_cost})")
                else:
                    unaffordable_items.append((item_config, item_cost))
                    logging.warning(f"[{self.id}] Cannot afford {item_name} (cost: ~{item_cost}, have: {coin_count})")
            
            # Check if we can afford everything
            if len(unaffordable_items) == 0:
                logging.info(f"[{self.id}] Can afford all items (total cost: ~{total_cost})")
                self.coins_checked = True
                self.set_phase("BUY_ITEMS")
                return self.loop_interval_ms
            
            # Try to find fallback items for unaffordable equipment
            fallback_found = False
            for item_config, item_cost in unaffordable_items:
                item_name = item_config["name"]
                
                # Check if this is equipment that might have fallbacks
                if self._is_equipment_item(item_name):
                    fallback_item = self._find_fallback_equipment(item_name)
                    if fallback_item:
                        logging.info(f"[{self.id}] Found fallback for {item_name}: {fallback_item}")
                        self.fallback_items[item_name] = fallback_item
                        fallback_found = True
                    else:
                        logging.warning(f"[{self.id}] No fallback found for {item_name}")
                else:
                    logging.warning(f"[{self.id}] No fallback available for {item_name} (not equipment)")
            
            if fallback_found:
                logging.info(f"[{self.id}] Found fallback items, updating purchase list")
                # Update items_to_buy with fallbacks
                self._update_items_with_fallbacks()
                self.coins_checked = True
                self.set_phase("BUY_ITEMS")
                return self.loop_interval_ms
            else:
                logging.error(f"[{self.id}] Cannot afford required items and no fallbacks available")
                self.error_message = f"Insufficient funds for required items. Need ~{total_cost} coins, have {coin_count}"
                return self.INSUFFICIENT_FUNDS
                
        except Exception as e:
            logging.error(f"[{self.id}] Error checking coins: {e}")
            self.error_message = f"Error checking coins: {e}"
            return self.ERROR
    
    def _handle_sell_items(self) -> int:
        """Handle selling items at the Grand Exchange."""
        logging.info(f"[{self.id}] Starting to sell items at GE...")
        
        # Handle regular quantity-based selling
        if self.sell_items:
            result = check_and_sell_required_items(
                self.sell_items, 
                self.sell_requirements, 
                self.id
            )
            
            if result["status"] == "complete":
                logging.info(f"[{self.id}] All items sold successfully!")
                self.selling_complete = True
                # After selling, check coins for buying if we have items to buy
                if self.items_to_buy:
                    self.set_phase("CHECK_COINS")
                else:
                    self.set_phase("CLOSE_GE")
                return self.loop_interval_ms
            
            elif result["status"] == "selling":
                logging.debug(f"[{self.id}] Still selling items...")
                return self.SELLING
            
            elif result["status"] == "error":
                error_msg = result.get("error", "Unknown error")
                logging.error(f"[{self.id}] Error selling items: {error_msg}")

                self.error_message = f"Error selling items: {error_msg}"
                return self.ERROR
            
            else:
                logging.warning(f"[{self.id}] Unknown result status: {result.get('status')}")
                return self.WAITING
        else:
            # No regular items to sell, just "sell all" items
            logging.info(f"[{self.id}] All items sold successfully!")
            self.selling_complete = True
            # After selling, check coins for buying if we have items to buy
            if self.items_to_buy:
                self.set_phase("CHECK_COINS")
            else:
                self.set_phase("CLOSE_GE")
            return self.loop_interval_ms
    
    def _is_equipment_item(self, item_name: str) -> bool:
        """Check if an item is equipment that might have fallbacks."""
        equipment_keywords = [
            "axe", "pickaxe", "scimitar", "sword", "dagger", "mace", "warhammer",
            "helmet", "helm", "platebody", "platelegs", "kiteshield", "shield",
            "gloves", "boots", "amulet", "ring", "cape", "armour", "armor"
        ]
        
        item_lower = item_name.lower()
        return any(keyword in item_lower for keyword in equipment_keywords)
    
    def _find_fallback_equipment(self, item_name: str) -> str:
        """Find a fallback equipment item that we can afford."""
        try:
            from ...actions import bank, inventory, equipment
            
            # Define equipment tiers (from worst to best)
            equipment_tiers = {
                "axe": ["Bronze axe", "Iron axe", "Steel axe", "Black axe", "Mithril axe", "Adamant axe", "Rune axe", "Dragon axe"],
                "scimitar": ["Bronze scimitar", "Iron scimitar", "Steel scimitar", "Black scimitar", "Mithril scimitar", "Adamant scimitar", "Rune scimitar"],
                "helmet": ["Bronze full helm", "Iron full helm", "Steel full helm", "Black full helm", "Mithril full helm", "Adamant full helm", "Rune full helm"],
                "platebody": ["Bronze platebody", "Iron platebody", "Steel platebody", "Black platebody", "Mithril platebody", "Adamant platebody", "Rune platebody"],
                "platelegs": ["Bronze platelegs", "Iron platelegs", "Steel platelegs", "Black platelegs", "Mithril platelegs", "Adamant platelegs", "Rune platelegs"],
                "kiteshield": ["Bronze kiteshield", "Iron kiteshield", "Steel kiteshield", "Black kiteshield", "Mithril kiteshield", "Adamant kiteshield", "Rune kiteshield"]
            }
            
            # Find the category for this item
            item_lower = item_name.lower()
            category = None
            for cat, items in equipment_tiers.items():
                if any(item.lower() in item_lower for item in items):
                    category = cat
                    break
            
            if not category:
                return None
            
            # Find the best item we can afford in this category
            target_items = equipment_tiers[category]
            current_index = -1
            
            # Find current item index
            for i, tier_item in enumerate(target_items):
                if tier_item.lower() in item_lower:
                    current_index = i
                    break
            
            if current_index == -1:
                return None
            
            # Helper function to check if item exists in any location
            def has_item_anywhere(item_name):
                return (bank.has_item(item_name) or 
                        inventory.has_item(item_name) or 
                        equipment.has_equipped(item_name))
            
            # Check if we have any lower-tier items in any location
            for i in range(current_index):
                fallback_item = target_items[i]
                if has_item_anywhere(fallback_item):
                    # Check which location has the item
                    location = "unknown"
                    if bank.has_item(fallback_item):
                        location = "bank"
                    elif inventory.has_item(fallback_item):
                        location = "inventory"
                    elif equipment.has_equipped(fallback_item):
                        location = "equipment"
                    
                    logging.info(f"[{self.id}] Found fallback {fallback_item} in {location}")
                    return fallback_item
            
            return None
            
        except Exception as e:
            logging.warning(f"[{self.id}] Error finding fallback equipment: {e}")
            return None
    
    def _update_items_with_fallbacks(self):
        """Update the items_to_buy list with fallback items."""
        updated_items = []
        
        for item_config in self.items_to_buy:
            item_name = item_config["name"]
            
            if item_name in self.fallback_items:
                # Replace with fallback item
                fallback_item = self.fallback_items[item_name]
                fallback_config = item_config.copy()
                fallback_config["name"] = fallback_item
                updated_items.append(fallback_config)
                logging.info(f"[{self.id}] Replacing {item_name} with fallback {fallback_item}")
            else:
                # Keep original item
                updated_items.append(item_config)
        
        # Update the internal lists
        self.items_to_buy = updated_items
        self.required_items = [item["name"] for item in updated_items]
        self.item_requirements = {}
        
        for item_config in updated_items:
            item_name = item_config["name"]
            self.item_requirements[item_name] = (
                item_config["quantity"],
                item_config["bumps"],
                item_config["set_price"]
            )
    
    def _handle_buy_items(self) -> int:
        """Handle buying items at the Grand Exchange."""
        logging.info(f"[{self.id}] Starting to buy items at GE...")
        
        # Use the reusable method from the ge actions
        result = check_and_buy_required_items(
            self.required_items, 
            self.item_requirements, 
            self.id
        )
        
        if result["status"] == "complete":
            logging.info(f"[{self.id}] All items purchased successfully!")
            self.set_phase("CLOSE_GE")
            return self.loop_interval_ms
        
        elif result["status"] == "buying":
            logging.debug(f"[{self.id}] Still buying items...")
            return self.BUYING
        
        elif result["status"] == "error":
            error_msg = result.get("error", "Unknown error")
            logging.error(f"[{self.id}] Error buying items: {error_msg}")

            self.error_message = f"Error buying items: {error_msg}"
            return self.ERROR
        
        else:
            logging.warning(f"[{self.id}] Unknown result status: {result.get('status')}")
            return self.WAITING
    
    def _handle_close_ge(self) -> int:
        """Handle closing the Grand Exchange interface."""
        logging.info(f"[{self.id}] Closing Grand Exchange interface...")
        
        try:
            # Use the close_ge function from the ge actions
            close_ge()
            logging.info(f"[{self.id}] Grand Exchange closed successfully")
            self.set_phase("COMPLETE")
            return self.loop_interval_ms
        except Exception as e:
            logging.warning(f"[{self.id}] Error closing GE: {e}")
            # Even if closing fails, we can still consider it complete
            # since the items were purchased successfully
            self.set_phase("COMPLETE")
            return self.loop_interval_ms
    
    def is_complete(self) -> bool:
        """Check if the GE buying is complete."""
        return self.buying_complete
    
    def get_error_message(self) -> str:
        """Get the current error message."""
        return self.error_message
    
    def reset(self):
        """Reset the plan to initial state."""
        self.state = {"phase": "TRAVEL_TO_GE"}
        self.buying_complete = False
        self.selling_complete = False
        self.error_message = None
        self.coins_checked = False
        self.fallback_items = {}
        logging.info(f"[{self.id}] Plan reset to initial state")
    
    def add_item(self, name: str, quantity: int, bumps: int = 0, set_price: int = 0):
        """Add an item to the buying list."""
        item_config = {
            "name": name,
            "quantity": quantity,
            "bumps": bumps,
            "set_price": set_price
        }
        
        self.items_to_buy.append(item_config)
        self.required_items.append(name)
        self.item_requirements[name] = (quantity, bumps, set_price)
        
        logging.info(f"[{self.id}] Added item to buy: {name} x{quantity}")
    
    def add_sell_item(self, name: str, quantity: int, bumps: int = 0, set_price: int = 0):
        """Add an item to the selling list."""
        item_config = {
            "name": name,
            "quantity": quantity,
            "bumps": bumps,
            "set_price": set_price
        }
        
        self.items_to_sell.append(item_config)

        self.sell_items.append(name)
        self.sell_requirements[name] = (quantity, bumps, set_price)
        logging.info(f"[{self.id}] Added item to sell: {name} x{quantity}")
    
    def clear_items(self):
        """Clear all items from the buying list."""
        self.items_to_buy.clear()
        self.required_items.clear()
        self.item_requirements.clear()
        logging.info(f"[{self.id}] Cleared all items from buying list")
    
    def clear_sell_items(self):
        """Clear all items from the selling list."""
        self.items_to_sell.clear()
        self.sell_items.clear()
        self.sell_requirements.clear()
        logging.info(f"[{self.id}] Cleared all items from selling list")


# Helper functions for easy setup
def create_ge_plan(items_to_buy: List[Dict[str, Any]] = None, items_to_sell: List[Dict[str, Any]] = None) -> GePlan:
    """
    Create a GE plan with lists of items to buy and/or sell.
    
    Args:
        items_to_buy: List of item configurations to buy
        items_to_sell: List of item configurations to sell
        
    Returns:
        Configured GePlan instance
        
    Example:
        buy_items = [
            {"name": "Trout", "quantity": 50, "bumps": 5, "set_price": 0},
            {"name": "Bronze scimitar", "quantity": 1, "bumps": 0, "set_price": 1000}
        ]
        sell_items = [
            {"name": "Raw beef", "quantity": 100, "bumps": 0, "set_price": 0}
        ]
        ge_plan = create_ge_plan(buy_items, sell_items)
    """
    return GePlan(items_to_buy, items_to_sell)


def create_simple_ge_plan(item_names: List[str], quantities: List[int] = None) -> GePlan:
    """
    Create a simple GE plan with default settings for buying.
    
    Args:
        item_names: List of item names to buy
        quantities: List of quantities (defaults to 1 for each item)
        
    Returns:
        Configured GePlan instance
        
    Example:
        ge_plan = create_simple_ge_plan(["Trout", "Bronze scimitar"], [50, 1])
    """
    if quantities is None:
        quantities = [1] * len(item_names)
    
    items_to_buy = []
    for name, quantity in zip(item_names, quantities):
        items_to_buy.append({
            "name": name,
            "quantity": quantity,
            "bumps": 5,  # Default bumps
            "set_price": 0  # Use bumps by default
        })
    
    return GePlan(items_to_buy)


def create_simple_sell_plan(item_names: List[str], quantities: List[int] = None) -> GePlan:
    """
    Create a simple GE plan with default settings for selling.
    
    Args:
        item_names: List of item names to sell
        quantities: List of quantities (defaults to 1 for each item)
        
    Returns:
        Configured GePlan instance
        
    Example:
        ge_plan = create_simple_sell_plan(["Raw beef", "Cowhide"], [100, 50])
    """
    if quantities is None:
        quantities = [1] * len(item_names)
    
    items_to_sell = []
    for name, quantity in zip(item_names, quantities):
        items_to_sell.append({
            "name": name,
            "quantity": quantity,
            "bumps": 0,  # Default no bumps for selling
            "set_price": 0  # Use market price by default
        })
    
    return GePlan(items_to_sell=items_to_sell)
