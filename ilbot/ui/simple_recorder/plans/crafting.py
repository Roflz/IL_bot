#!/usr/bin/env python3
"""
Crafting Plan
=============

Basic crafting plan that cycles between Edgeville bank and crafting.
Uses BankPlan for banking and has a CRAFTING phase for the actual crafting.

Features:
- Uses BankPlan for character setup and banking
- Cycles between banking and crafting phases
- Handles moulds, gold bars, and gems
"""

import time
import logging
from pathlib import Path

# Add the parent directory to the path for imports
import sys

import requests

from ..actions.travel import in_area, go_to
from ..helpers.utils import press_spacebar
from ..helpers.widgets import widget_exists

sys.path.insert(0, str(Path(__file__).parent.parent))

from .base import Plan
from .utilities.bank_plan import BankPlan
from .utilities.ge import GePlan, create_ge_plan
from ..actions import objects, player, inventory, bank, travel, chat, widgets, equipment
from ..actions.player import logged_in
from ..actions.timing import wait_until
from ..constants import BANK_REGIONS, EXPERIENCE_TABLE, CRAFTING_EXP


class CraftingPlan(Plan):
    """Crafting plan using BankPlan for banking and custom crafting logic."""
    
    id = "CRAFTING"
    label = "Crafting: Edgeville Bank"
    
    def __init__(self):
        self.state = {"phase": "BANK"}
        self.next = self.state["phase"]
        self.loop_interval_ms = 600
        
        # Set up camera immediately during initialization
        try:
            from ..helpers.camera import setup_camera_optimal
            setup_camera_optimal()
        except Exception as e:
            logging.warning(f"[{self.id}] Could not setup camera: {e}")
        
        # Configuration
        self.bank_area = None
        self.crafting_items = {
            "gold ring": (28, "ring mould", "sapphire"),  # (quantity, mould, gem)
            "gold necklace": (28, "necklace mould", "emerald")
        }
        
        # Create bank plan for banking
        self.bank_plan = BankPlan(
            bank_area=self.bank_area,
            food_item=None,  # No food for crafting
            food_quantity=0,
            equipment_config={
                "weapon_tiers": [],  # No weapons needed
                "armor_tiers": {},   # No armor needed
                "jewelry_tiers": {}, # No jewelry needed
                "tool_tiers": []     # No tools needed
            },
            inventory_config={
                "required_items": [],  # Will be set dynamically based on crafting level
                "optional_items": [],
                "deposit_all": True
            }
        )
        
        # GE strategy configuration
        self.ge_strategy = {
            # Mould strategies
            "ring mould": {"quantity": 1, "bumps": 0, "set_price": 1000},
            "necklace mould": {"quantity": 1, "bumps": 0, "set_price": 1000},
            "bracelet mould": {"quantity": 1, "bumps": 0, "set_price": 1000},
            "amulet_mould": {"quantity": 1, "bumps": 0, "set_price": 1000},
            
            # Gem strategies
            "sapphire": {"quantity": 28, "bumps": 5, "set_price": 0},
            "emerald": {"quantity": 28, "bumps": 5, "set_price": 0},
            "ruby": {"quantity": 28, "bumps": 5, "set_price": 0},
            "diamond": {"quantity": 28, "bumps": 5, "set_price": 0},
            
            # Gold bar strategy
            "gold bar": {"quantity": 28, "bumps": 5, "set_price": 0},
            
            # Default strategy
            "default": {"quantity": 1, "bumps": 5, "set_price": 0}
        }
        
        # State tracking
        self.ge_plan = None  # Will be created when needed
        
        logging.info(f"[{self.id}] Plan initialized")
        logging.info(f"[{self.id}] Using bank plan for character setup")
        logging.info(f"[{self.id}] Bank area: {self.bank_area}")
        logging.info(f"[{self.id}] Crafting items: {self.crafting_items}")
        logging.info(f"[{self.id}] GE strategy: {self.ge_strategy}")
    
    def set_phase(self, phase: str, camera_setup: bool = True):
        """Set the current phase."""
        from ..helpers.phase_utils import set_phase_with_camera
        return set_phase_with_camera(self, phase, camera_setup)
    
    def loop(self, ui) -> int:
        """Main loop method."""
        phase = self.state.get("phase", "BANK")
        logged_in = player.logged_in()
        if not logged_in:
            player.login()
            return self.loop_interval_ms

        match(phase):
            case "BANK":
                return self._handle_bank(ui)

            case "MISSING_ITEMS":
                return self._handle_missing_items(ui)

            case "CRAFTING":
                return self._handle_crafting()

        logging.warning(f"[{self.id}] Unknown phase: {phase}")
        return self.loop_interval_ms
    
    def _handle_bank(self, ui) -> int:
        """Handle banking phase - delegate all banking logic to bank plan."""
        # Update bank plan inventory config based on crafting level
        self._update_bank_plan_for_crafting_level()
        
        bank_status = self.bank_plan.loop(ui)
        
        if bank_status == BankPlan.SUCCESS:
            logging.info(f"[{self.id}] Banking completed successfully!")
            self.set_phase("CRAFTING")
            return self.loop_interval_ms
        
        elif bank_status == BankPlan.MISSING_ITEMS:
            error_msg = self.bank_plan.get_error_message()
            logging.warning(f"[{self.id}] Banking failed - missing items: {error_msg}")
            logging.warning(f"[{self.id}] Will try to buy missing items from GE")
            self.set_phase("MISSING_ITEMS")
            return self.loop_interval_ms
        
        else:
            # Still working on banking (TRAVELING, BANKING, EQUIPPING, etc.)
            # Return the bank plan's status so it can continue working
            return bank_status
    
    def _update_bank_plan_for_crafting_level(self):
        """Update bank plan inventory config based on current crafting level."""
        try:
            # Get current crafting level
            crafting_level = player.get_skill_level('crafting')
            logging.info(f"[{self.id}] Current crafting level: {crafting_level}")
            
            if crafting_level < 5:
                # Level 1-4: Leather crafting - calculate how much leather needed
                leather_needed = self._calculate_leather_needed_for_level_5()
                logging.info(f"[{self.id}] Setting up for leather crafting (level {crafting_level}) - need {leather_needed} leather")
                self.bank_plan.inventory_config["required_items"] = [
                    {"name": "Needle", "quantity": 1},
                    {"name": "Thread", "quantity": 10},
                    {"name": "Leather", "quantity": leather_needed}
                ]
                
            elif crafting_level < 20:
                # Level 5-19: Gold ring crafting - calculate how many rings needed
                rings_needed = self._calculate_gold_rings_needed_for_level_20()
                logging.info(f"[{self.id}] Setting up for gold ring crafting (level {crafting_level}) - need {rings_needed} gold bars")
                self.bank_plan.inventory_config["required_items"] = [
                    {"name": "ring mould", "quantity": 1},
                    {"name": "Gold bar", "quantity": rings_needed}
                ]
                
            else:
                # Level 20+: Sapphire ring crafting - calculate how many rings needed
                self.bank_plan.inventory_config["required_items"] = [
                    {"name": "ring mould", "quantity": 1},
                    {"name": "Sapphire", "quantity": 13},
                    {"name": "Gold bar", "quantity": 13}
                ]
                
        except Exception as e:
            logging.error(f"[{self.id}] Error updating bank plan for crafting level: {e}")
            # Default to basic gold ring setup
            self.bank_plan.inventory_config["required_items"] = [
                {"name": "Gold bar", "quantity": 28},
                {"name": "ring mould", "quantity": 1}
            ]
    
    def _get_crafting_level(self) -> int:
        """Get current crafting level from player stats."""
        try:
            from ..actions.player import get_player_stats
            stats = get_player_stats()
            if stats and "Crafting" in stats:
                return int(stats["Crafting"].get("level", 1))
            return 1  # Default to level 1 if can't get stats
        except Exception as e:
            logging.error(f"[{self.id}] Error getting crafting level: {e}")
            return 1  # Default to level 1
    
    
    def _calculate_leather_needed_for_level_5(self) -> int:
        """Calculate how much leather is needed to reach level 5."""
        try:
            # Get current crafting level and experience
            current_exp = player.get_skill_xp('crafting')
            
            # Calculate experience needed to reach level 5
            exp_needed_for_level_5 = EXPERIENCE_TABLE[5] - current_exp
            
            # Each leather gives 13.8 experience (leather gloves)
            exp_per_leather = CRAFTING_EXP["leather_gloves"]
            
            # Calculate leather needed (round up)
            leather_needed = int(exp_needed_for_level_5 / exp_per_leather) + 1
            
            # Cap at 26 (inventory space minus needle and thread)
            leather_needed = min(leather_needed, 26)
            
            logging.info(f"[{self.id}] Current exp: {current_exp}, Need for level 5: {EXPERIENCE_TABLE[5]}, Exp needed: {exp_needed_for_level_5}")
            logging.info(f"[{self.id}] Leather needed: {leather_needed} (each gives {exp_per_leather} exp)")
            
            return leather_needed
            
        except Exception as e:
            logging.error(f"[{self.id}] Error calculating leather needed: {e}")
            # Default to 26 leather if calculation fails
            return 26
    
    def _get_ge_items_for_crafting_level(self) -> tuple:
        """Get GE items to buy and sell based on current crafting level."""
        try:
            # Get current crafting level
            crafting_level = player.get_skill_level('crafting')
            logging.info(f"[{self.id}] Current crafting level: {crafting_level}")
            
            if crafting_level < 5:
                # Level 1-4: Buy leather, needle, thread for leather crafting
                leather_needed = self._calculate_leather_needed_for_ge()
                logging.info(f"[{self.id}] Setting up GE for leather crafting (level {crafting_level}) - need {leather_needed} leather")
                items_to_buy = [
                    {"name": "Needle", "quantity": 1, "bumps": 0, "set_price": 500},
                    {"name": "Thread", "quantity": 10, "bumps": 10, "set_price": 0},
                    {"name": "Leather", "quantity": leather_needed, "bumps": 5, "set_price": 0}
                ]
                items_to_sell = []  # No items to sell at low levels
                return items_to_buy, items_to_sell
                
            elif crafting_level < 20:
                # Level 5-19: Buy gold bars and ring mould for gold ring crafting
                rings_needed = self._calculate_gold_rings_needed_for_level_20()
                logging.info(f"[{self.id}] Setting up GE for gold ring crafting (level {crafting_level}) - need {rings_needed} gold bars")
                items_to_buy = [
                    {"name": "Gold bar", "quantity": rings_needed, "bumps": 5, "set_price": 0},
                    {"name": "ring mould", "quantity": 1, "bumps": 0, "set_price": 1000}
                ]
                items_to_sell = []  # No items to sell at mid levels
                return items_to_buy, items_to_sell
                
            else:
                # Level 20+: Dynamic GE plan - sell all rings, buy based on available funds
                logging.info(f"[{self.id}] Setting up dynamic GE for sapphire ring crafting (level {crafting_level})")
                
                # Get bank state and deposit inventory
                bank_coins, gold_rings, sapphire_rings = self._deposit_inventory_and_count_bank()
                
                # Get current GE prices
                prices = self._get_current_ge_prices(["Gold ring", "Sapphire ring", "Sapphire", "Gold bar"])
                
                # Calculate sell proceeds with bumps
                gold_ring_sell_price = self._calculate_sell_price(prices["Gold ring"], bumps=5)
                sapphire_ring_sell_price = self._calculate_sell_price(prices["Sapphire ring"], bumps=5)
                sell_proceeds = (gold_rings * gold_ring_sell_price) + (sapphire_rings * sapphire_ring_sell_price)
                
                # Calculate spending budget (80% of total funds)
                budget = self._calculate_spending_budget(bank_coins, sell_proceeds)
                
                # Calculate buy prices with bumps
                sapphire_buy_price = self._calculate_buy_price(prices["Sapphire"], bumps=5)
                gold_bar_buy_price = self._calculate_buy_price(prices["Gold bar"], bumps=5)
                
                # Calculate equal purchase quantities
                sapphire_qty, gold_bar_qty = self._calculate_equal_purchase_quantities(
                    budget, sapphire_buy_price, gold_bar_buy_price
                )
                
                # Items to sell (only if we have them)
                items_to_sell = []
                
                # Check if we have gold rings to sell
                if gold_rings > 0:
                    items_to_sell.append({
                        "name": "Gold ring", 
                        "quantity": -1, 
                        "bumps": 5, 
                        "set_price": 0
                    })
                    logging.info(f"[{self.id}] Will sell {gold_rings} gold rings")
                
                # Check if we have sapphire rings to sell
                if sapphire_rings > 0:
                    items_to_sell.append({
                        "name": "Sapphire ring", 
                        "quantity": -1, 
                        "bumps": 5, 
                        "set_price": 0
                    })
                    logging.info(f"[{self.id}] Will sell {sapphire_rings} sapphire rings")
                
                # Items to buy (calculated quantities)
                items_to_buy = [
                    {"name": "Sapphire", "quantity": sapphire_qty, "bumps": 5, "set_price": 0},
                    {"name": "Gold bar", "quantity": gold_bar_qty, "bumps": 5, "set_price": 0}
                ]
                
                # Only add ring mould if we don't have one
                if not bank.has_item("ring mould") and not inventory.has_item("ring mould") and not equipment.has_equipped("ring mould"):
                    items_to_buy.append({
                        "name": "ring mould", 
                        "quantity": 1, 
                        "bumps": 0, 
                        "set_price": 1000
                    })
                    logging.info(f"[{self.id}] Will buy ring mould (don't have one)")
                else:
                    logging.info(f"[{self.id}] Already have ring mould, skipping purchase")
                
                return items_to_buy, items_to_sell
                
        except Exception as e:
            logging.error(f"[{self.id}] Error getting GE items for crafting level: {e}")
            # Default to basic gold ring setup
            items_to_buy = [
                {"name": "Gold bar", "quantity": 28, "bumps": 0, "set_price": 0},
                {"name": "ring mould", "quantity": 1, "bumps": 0, "set_price": 0}
            ]
            items_to_sell = []
            return items_to_buy, items_to_sell
    
    def _calculate_leather_needed_for_ge(self) -> int:
        """Calculate how much leather is needed to reach level 5 for GE purchases."""
        try:
            # Get current crafting level and experience
            current_exp = player.get_skill_xp('crafting')
            
            # Calculate experience needed to reach level 5
            exp_needed_for_level_5 = EXPERIENCE_TABLE[5] - current_exp
            
            # Each leather gives 13.8 experience (leather gloves)
            exp_per_leather = CRAFTING_EXP["leather_gloves"]
            
            # Calculate leather needed (round up)
            leather_needed = int(exp_needed_for_level_5 / exp_per_leather) + 1
            
            # No cap for GE plan - can buy as much as needed
            logging.info(f"[{self.id}] Current exp: {current_exp}, Need for level 5: {EXPERIENCE_TABLE[5]}, Exp needed: {exp_needed_for_level_5}")
            logging.info(f"[{self.id}] Leather needed for GE: {leather_needed} (each gives {exp_per_leather} exp)")
            
            return leather_needed
            
        except Exception as e:
            logging.error(f"[{self.id}] Error calculating leather needed for GE: {e}")
            # Default to 50 leather if calculation fails
            return 50
    
    def _calculate_gold_rings_needed_for_level_20(self) -> int:
        """Calculate how many gold rings are needed to reach level 20."""
        try:
            # Get current crafting level and experience
            current_exp = player.get_skill_xp('crafting')
            
            # Calculate experience needed to reach level 20
            exp_needed_for_level_20 = EXPERIENCE_TABLE[20] - current_exp
            
            # Each gold ring gives 15 experience
            exp_per_ring = CRAFTING_EXP["gold_ring"]
            
            # Calculate rings needed (round up)
            rings_needed = int(exp_needed_for_level_20 / exp_per_ring) + 1
            
            logging.info(f"[{self.id}] Current exp: {current_exp}, Need for level 20: {EXPERIENCE_TABLE[20]}, Exp needed: {exp_needed_for_level_20}")
            logging.info(f"[{self.id}] Gold rings needed: {rings_needed} (each gives {exp_per_ring} exp)")
            
            return rings_needed
            
        except Exception as e:
            logging.error(f"[{self.id}] Error calculating gold rings needed: {e}")
            # Default to 27 rings if calculation fails
            return 27
    
    def _deposit_inventory_and_count_bank(self):
        """Deposit inventory and count bank contents for level 20+ crafting."""
        try:
            # Deposit all inventory
            bank.deposit_inventory()
            if not wait_until(inventory.is_empty, max_wait_ms=3000):
                return self.loop_interval_ms
            
            # Count coins in bank
            bank_coins = bank.get_item_count("Coins")
            
            # Count gold rings and sapphire rings in bank
            gold_rings = bank.get_item_count("Gold ring")
            sapphire_rings = bank.get_item_count("Sapphire ring")
            
            logging.info(f"[{self.id}] Bank state - Coins: {bank_coins}, Gold rings: {gold_rings}, Sapphire rings: {sapphire_rings}")
            
            return bank_coins, gold_rings, sapphire_rings
            
        except Exception as e:
            logging.error(f"[{self.id}] Error depositing inventory and counting bank: {e}")
            return 0, 0, 0
    
    def _get_current_ge_prices(self, items):
        """Get current GE prices for items via Weird Gloop API."""
        try:
            import requests
            import json
            
            # Weird Gloop API endpoint for latest prices
            base_url = "https://api.weirdgloop.org/exchange/history/osrs/latest"
            
            # Build query string with item names
            item_names = "|".join(items)
            url = f"{base_url}?name={item_names}"
            
            logging.info(f"[{self.id}] Fetching GE prices from Weird Gloop API: {url}")
            
            # Make API request
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # Check if API returned an error
                if not data.get("success", True):
                    error_msg = data.get("error", "Unknown error")
                    logging.warning(f"[{self.id}] API error: {error_msg}")
                    return self._get_fallback_prices(items)
                
                # Extract prices from API response
                prices = {}
                for item in items:
                    # Look for item in API response
                    found = False
                    for name, item_data in data.items():
                        if isinstance(item_data, dict):
                            if item.lower() == name.lower():
                                price = item_data.get("price", 0)
                                if price > 0:
                                    prices[item] = price
                                    found = True
                                    logging.info(f"[{self.id}] Found {item} price from API: {price}")
                                    break

                    if not found:
                        # Use fallback price
                        fallback_prices = self._get_fallback_prices([item])
                        prices[item] = fallback_prices[item]
                        logging.warning(f"[{self.id}] Using fallback price for {item}: {prices[item]}")
                
                logging.info(f"[{self.id}] Final prices from Weird Gloop API: {prices}")
                return prices
            else:
                logging.warning(f"[{self.id}] API request failed with status {response.status_code}")
                return self._get_fallback_prices(items)
            
        except Exception as e:
            logging.error(f"[{self.id}] Error getting GE prices from Weird Gloop API: {e}")
            return self._get_fallback_prices(items)
    
    def _get_fallback_prices(self, items):
        """Get fallback prices for items."""
        fallback_prices = {
            "Gold ring": 100,
            "Sapphire ring": 400,
            "Sapphire": 200,
            "Gold bar": 100
        }
        return {item: fallback_prices.get(item, 100) for item in items}
    
    
    def _calculate_sell_price(self, base_price, bumps):
        """Calculate sell price with -5% per bump (compounding)."""
        return int(base_price * (0.95 ** bumps))
    
    def _calculate_buy_price(self, base_price, bumps):
        """Calculate buy price with +5% per bump (compounding)."""
        return int(base_price * (1.05 ** bumps))
    
    def _calculate_spending_budget(self, current_coins, sell_proceeds):
        """Calculate 80% of total available funds."""
        total_funds = current_coins + sell_proceeds
        budget = int(total_funds * 0.8)
        logging.info(f"[{self.id}] Budget calculation - Current coins: {current_coins}, Sell proceeds: {sell_proceeds}, Total: {total_funds}, Budget (80%): {budget}")
        return budget
    
    def _calculate_equal_purchase_quantities(self, budget, sapphire_price, gold_bar_price):
        """Calculate equal quantities of sapphires and gold bars."""
        total_cost_per_pair = sapphire_price + gold_bar_price
        max_pairs = budget // total_cost_per_pair
        
        sapphire_qty = max_pairs
        gold_bar_qty = max_pairs
        
        logging.info(f"[{self.id}] Purchase calculation - Budget: {budget}, Cost per pair: {total_cost_per_pair}, Max pairs: {max_pairs}")
        logging.info(f"[{self.id}] Quantities - Sapphires: {sapphire_qty}, Gold bars: {gold_bar_qty}")
        
        return sapphire_qty, gold_bar_qty
    
    def _handle_missing_items(self, ui) -> int:
        """Handle missing items phase by using GE utility."""
        # If we don't have a GE plan yet, create one with the missing items
        if self.ge_plan is None:
            error_msg = self.bank_plan.get_error_message()
            logging.info(f"[{self.id}] Creating GE plan for missing items: {error_msg}")
            
            # Create GE plan based on current crafting level
            if player.get_skill_level("crafting") >= 20 and not bank.is_open():
                bank.open_bank()

            items_to_buy, items_to_sell = self._get_ge_items_for_crafting_level()

            # Create GE plan based on current crafting level
            if bank.is_open():
                bank.close_bank()
            
            self.ge_plan = GePlan(items_to_buy=items_to_buy, items_to_sell=items_to_sell)
            logging.info(f"[{self.id}] Created GE plan to buy: {[item['name'] for item in items_to_buy]}")
        
        # Use the GE plan to buy missing items
        ge_status = self.ge_plan.loop(ui)
        
        if ge_status == GePlan.SUCCESS:
            logging.info(f"[{self.id}] Successfully purchased all missing items!")
            # Reset bank plan and try banking again
            self.bank_plan.reset()
            self.ge_plan = None  # Clear GE plan
            self.set_phase("BANK")
            return self.loop_interval_ms
        
        else:
            # Still working on buying items (TRAVELING, CHECKING_COINS, BUYING, WAITING)
            logging.info(f"[{self.id}] GE plan in progress... Status: {ge_status}")
            return ge_status
    
    def _handle_crafting(self) -> int:
        """Handle crafting phase."""
        if not objects.object_exists("furnace", 18) and inventory.has_item("gold bar"):
            go_to("EDGEVILLE_BANK")
            return self.loop_interval_ms

        if not inventory.has_item("gold bar") and not inventory.has_item("sapphire") and not inventory.has_item("leather"):
            # Reset bank plan so it can run through the banking process again
            self.bank_plan.reset()
            self.set_phase("BANK")
            return self.loop_interval_ms

        if chat.can_continue():
            chat.continue_dialogue()
            return self.loop_interval_ms

        if wait_until(lambda: player.get_player_animation() == "SEWING" or player.get_player_animation() == "SMELTING", max_wait_ms=1000):
            return 1000

        if inventory.has_item("leather"):
            inventory.use_item_on_item("needle", "leather")
            if not wait_until(lambda: widget_exists(17694720)):
                return self.loop_interval_ms
            press_spacebar()
            if not wait_until(lambda: player.get_player_animation() == 1249):
                return self.loop_interval_ms
            return self.loop_interval_ms

        if inventory.has_items({"gold bar": "any", "ring mould": 1}) and not inventory.has_item("Sapphire"):
            objects.click_object_closest_by_distance_simple("Furnace", "Smelt")
            if not wait_until(lambda: widget_exists(29229056)): # Crafting interface
                return self.loop_interval_ms
            time.sleep(0.5)
            opts = widgets.get_crafting_options_rings()
            for widget in opts:
                if widget.get('itemId') == 1635:
                    widgets.click_widget(widget.get("id"))
                    break

            if not wait_until(lambda: player.get_player_animation() == "SMELTING"):
                return self.loop_interval_ms
            return self.loop_interval_ms

        if inventory.has_items({"gold bar": "any", "ring mould": 1, "sapphire": "any"}):
            objects.click_object_closest_by_distance_simple("furnace", "smelt")
            if not wait_until(lambda: widget_exists(29229056)): # Crafting interface
                return self.loop_interval_ms
            opts = widgets.get_crafting_options_rings()
            for widget in opts:
                if widget.get('itemId') == 1637:
                    widgets.click_widget(widget.get("id"))
                    break

            if not wait_until(lambda: player.get_player_animation() == "SMELTING"):
                return self.loop_interval_ms
            return self.loop_interval_ms
        
        return self.loop_interval_ms
    
    def _handle_error(self) -> int:
        """Handle error state."""
        logging.error(f"[{self.id}] Plan is in error state")
        logging.error(f"[{self.id}] Check logs above for details")
        
        # Wait and let user see the error
        time.sleep(10)
        return self.loop_interval_ms
