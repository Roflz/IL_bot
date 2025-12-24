#!/usr/bin/env python3
"""
Crafting Helper Methods
======================

Helper methods for the crafting plan. Contains all the complex logic for:
- Optimal jewelry selection
- Material analysis
- GE price calculations
- Profit calculations
"""

import logging
import requests
from typing import List, Dict, Tuple

from actions import bank, inventory, player
from actions import wait_until
from constants import EXPERIENCE_TABLE, CRAFTING_EXP


class CraftingMethods:
    """Helper methods for crafting plan."""
    
    def __init__(self, plan_id: str):
        self.id = plan_id
    
    def get_crafting_level(self) -> int:
        """Get current crafting level from player stats."""
        try:
            from actions import get_player_stats
            stats = get_player_stats()
            if stats and "Crafting" in stats:
                return int(stats["Crafting"].get("level", 1))
            return 1  # Default to level 1 if can't get stats
        except Exception as e:
            logging.error(f"[{self.id}] Error getting crafting level: {e}")
            return 1  # Default to level 1
    
    def calculate_leather_needed_for_level_5(self) -> int:
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
    
    def calculate_gold_rings_needed_for_level_20(self) -> int:
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
    
    def calculate_gold_bracelets_needed_for_level_20(self) -> int:
        """Calculate how many gold bracelets are needed to reach level 20."""
        try:
            # Get current crafting level and experience
            current_exp = player.get_skill_xp('crafting')
            
            # Calculate experience needed to reach level 20
            exp_needed_for_level_20 = EXPERIENCE_TABLE[20] - current_exp
            
            # Each gold bracelet gives 11 experience
            exp_per_bracelet = CRAFTING_EXP.get("gold_bracelet", 11.0)
            
            # Calculate bracelets needed (round up)
            bracelets_needed = int(exp_needed_for_level_20 / exp_per_bracelet) + 1
            
            logging.info(f"[{self.id}] Current exp: {current_exp}, Need for level 20: {EXPERIENCE_TABLE[20]}, Exp needed: {exp_needed_for_level_20}")
            logging.info(f"[{self.id}] Gold bracelets needed: {bracelets_needed} (each gives {exp_per_bracelet} exp)")
            
            return bracelets_needed
            
        except Exception as e:
            logging.error(f"[{self.id}] Error calculating gold bracelets needed: {e}")
            # Default to 36 bracelets if calculation fails (more needed since less XP per bracelet)
            return 36
    
    def calculate_leather_needed_for_ge(self) -> int:
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
    
    def deposit_inventory_and_count_bank(self) -> Tuple[int, Dict]:
        """Deposit inventory and count bank contents for level 20+ crafting."""
        try:
            # Deposit all inventory
            bank.deposit_inventory()
            if not wait_until(inventory.is_empty, max_wait_ms=3000):
                return 0, {}
            
            # Count coins in bank
            bank_coins = bank.get_item_count("Coins")
            
            # Count all possible crafted items
            existing_items = {}
            possible_items = [
                "Gold ring", "Gold necklace", "Gold bracelet", "Gold amulet (u)", "Gold amulet",
                "Sapphire ring", "Sapphire necklace", "Sapphire bracelet", "Sapphire amulet (u)", "Sapphire amulet",
                "Emerald ring", "Emerald necklace", "Emerald bracelet", "Emerald amulet (u)", "Emerald amulet",
                "Ruby ring", "Ruby necklace"
            ]
            
            for item in possible_items:
                count = bank.get_item_count(item)
                if count > 0:
                    existing_items[item] = count
            
            logging.info(f"[{self.id}] Bank state: {bank_coins} coins, {len(existing_items)} item types")
            for item, count in existing_items.items():
                logging.info(f"[{self.id}] Bank has {count} {item}")
            
            return bank_coins, existing_items
            
        except Exception as e:
            logging.error(f"[{self.id}] Error depositing inventory and counting bank: {e}")
            return 0, {}
    
    def get_current_ge_prices(self, items: List[str]) -> Dict[str, int]:
        """Get current GE prices for items via Weird Gloop API."""
        try:
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
                    return self.get_fallback_prices(items)
                
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
                        fallback_prices = self.get_fallback_prices([item])
                        prices[item] = fallback_prices[item]
                        logging.warning(f"[{self.id}] Using fallback price for {item}: {prices[item]}")
                
                logging.info(f"[{self.id}] Final prices from Weird Gloop API: {prices}")
                return prices
            else:
                logging.warning(f"[{self.id}] API request failed with status {response.status_code}")
                return self.get_fallback_prices(items)
            
        except Exception as e:
            logging.error(f"[{self.id}] Error getting GE prices from Weird Gloop API: {e}")
            return self.get_fallback_prices(items)
    
    def get_fallback_prices(self, items: List[str]) -> Dict[str, int]:
        """Get fallback prices for items."""
        fallback_prices = {
            "Gold ring": 100,
            "Sapphire ring": 400,
            "Sapphire": 200,
            "Gold bar": 100
        }
        return {item: fallback_prices.get(item, 100) for item in items}
    
    def calculate_sell_price(self, base_price: int, bumps: int) -> int:
        """Calculate sell price with -5% per bump (compounding)."""
        return int(base_price * (0.95 ** bumps))
    
    def calculate_buy_price(self, base_price: int, bumps: int) -> int:
        """Calculate buy price with +5% per bump (compounding)."""
        return int(base_price * (1.05 ** bumps))
    
    def calculate_spending_budget(self, current_coins: int, sell_proceeds: int) -> int:
        """Calculate 80% of total available funds."""
        total_funds = current_coins + sell_proceeds
        logging.info(f"[{self.id}] Budget calculation - Current coins: {current_coins}, Sell proceeds: {sell_proceeds}, Funds: {total_funds}")
        return total_funds
    
    def calculate_equal_purchase_quantities(self, budget: int, sapphire_price: int, gold_bar_price: int) -> Tuple[int, int]:
        """Calculate equal quantities of sapphires and gold bars."""
        total_cost_per_pair = sapphire_price + gold_bar_price
        max_pairs = budget // total_cost_per_pair
        
        sapphire_qty = max_pairs
        gold_bar_qty = max_pairs
        
        logging.info(f"[{self.id}] Purchase calculation - Budget: {budget}, Cost per pair: {total_cost_per_pair}, Max pairs: {max_pairs}")
        logging.info(f"[{self.id}] Quantities - Sapphires: {sapphire_qty}, Gold bars: {gold_bar_qty}")
        
        return sapphire_qty, gold_bar_qty
    
    def check_bank_materials(self) -> Dict[str, int]:
        """Check what materials are available in the bank."""
        try:
            materials = {}
            
            # Check for gems and gold bars
            gem_types = ["Sapphire", "Emerald", "Ruby", "Diamond", "Opal", "Jade", "Topaz"]
            for gem in gem_types:
                count = bank.get_item_count(gem)
                if count > 0:
                    materials[gem] = count
                    logging.info(f"[{self.id}] Bank has {count} {gem}")
            
            # Check for gold bars
            gold_bars = bank.get_item_count("Gold bar")
            if gold_bars > 0:
                materials["Gold bar"] = gold_bars
                logging.info(f"[{self.id}] Bank has {gold_bars} Gold bar")
            
            return materials
            
        except Exception as e:
            logging.error(f"[{self.id}] Error checking bank materials: {e}")
            return {}
    
    def determine_optimal_crafting_item_from_available(self, available_jewelry: Dict, crafting_level: int) -> str:
        """Determine the most profitable jewelry from available options."""
        try:
            if not available_jewelry:
                return "Sapphire ring"  # Default fallback
            
            # Get all required materials for price checking
            all_materials = set()
            for item_data in available_jewelry.values():
                all_materials.update(item_data["materials"].keys())
            
            # Add all possible products for selling
            all_products = list(available_jewelry.keys())
            
            # Get current GE prices
            price_items = list(all_materials) + all_products
            prices = self.get_current_ge_prices(price_items)
            
            # Calculate profit for each available jewelry
            item_profits = {}
            for item_name, item_data in available_jewelry.items():
                try:
                    # Get sell price (with bumps)
                    sell_price = prices.get(item_name, 0)
                    
                    # Calculate material costs
                    material_cost = 0
                    for material, qty in item_data["materials"].items():
                        material_price = prices.get(material, 0)
                        material_cost += material_price * qty
                    
                    # Calculate profit per item
                    profit = sell_price - material_cost
                    profit_per_xp = profit / item_data["xp"] if item_data["xp"] > 0 else 0
                    
                    item_profits[item_name] = {
                        "profit": profit,
                        "profit_per_xp": profit_per_xp,
                        "sell_price": sell_price,
                        "material_cost": material_cost,
                        "xp": item_data["xp"],
                        "level": item_data["level"],
                        "materials": item_data["materials"]
                    }
                    
                    logging.info(f"[{self.id}] {item_name}: Profit={profit:.0f}gp, Profit/XP={profit_per_xp:.2f}gp, "
                               f"Sell={sell_price:.0f}gp, Cost={material_cost:.0f}gp")
                    
                except Exception as e:
                    logging.warning(f"[{self.id}] Error calculating profit for {item_name}: {e}")
                    continue
            
            if not item_profits:
                logging.error(f"[{self.id}] No profitable jewelry found")
                return list(available_jewelry.keys())[0]  # Return first available
            
            # Sort by profit per XP (most efficient first)
            sorted_items = sorted(item_profits.items(), 
                                key=lambda x: x[1]["profit"], reverse=True)
            
            # Choose the best item
            best_item_name, best_item_data = sorted_items[0]
            logging.info(f"[{self.id}] Selected best jewelry: {best_item_name} "
                        f"(Profit: {best_item_data['profit']:.2f}gp)")
            
            return best_item_name
            
        except Exception as e:
            logging.error(f"[{self.id}] Error determining optimal jewelry: {e}")
            return list(available_jewelry.keys())[0] if available_jewelry else "Sapphire ring"
    
    def get_optimal_materials_for_missing_items(self, crafting_level: int) -> List[Dict]:
        """Determine optimal materials to buy when we have no jewelry materials."""
        try:
            # Define jewelry options (no gold jewelry)
            jewelry_options = {
                # Rings
                "Sapphire ring": {"level": 20, "materials": {"Gold bar": 1, "Sapphire": 1}, "mould": "ring mould", "xp": 40},
                "Emerald ring": {"level": 27, "materials": {"Gold bar": 1, "Emerald": 1}, "mould": "ring mould", "xp": 55},
                "Ruby ring": {"level": 34, "materials": {"Gold bar": 1, "Ruby": 1}, "mould": "ring mould", "xp": 70},
                "Diamond ring": {"level": 43, "materials": {"Gold bar": 1, "Diamond": 1}, "mould": "ring mould", "xp": 85},
                "Opal ring": {"level": 16, "materials": {"Gold bar": 1, "Opal": 1}, "mould": "ring mould", "xp": 30},
                "Jade ring": {"level": 13, "materials": {"Gold bar": 1, "Jade": 1}, "mould": "ring mould", "xp": 32},
                "Topaz ring": {"level": 16, "materials": {"Gold bar": 1, "Topaz": 1}, "mould": "ring mould", "xp": 35},
                # Necklaces
                "Sapphire necklace": {"level": 22, "materials": {"Gold bar": 1, "Sapphire": 1}, "mould": "necklace mould", "xp": 55},
                "Emerald necklace": {"level": 29, "materials": {"Gold bar": 1, "Emerald": 1}, "mould": "necklace mould", "xp": 60},
                "Ruby necklace": {"level": 40, "materials": {"Gold bar": 1, "Ruby": 1}, "mould": "necklace mould", "xp": 75},
                "Diamond necklace": {"level": 56, "materials": {"Gold bar": 1, "Diamond": 1}, "mould": "necklace mould", "xp": 90},
                # Bracelets
                "Sapphire bracelet": {"level": 23, "materials": {"Gold bar": 1, "Sapphire": 1}, "mould": "bracelet mould", "xp": 60},
                "Emerald bracelet": {"level": 30, "materials": {"Gold bar": 1, "Emerald": 1}, "mould": "bracelet mould", "xp": 65},
                "Ruby bracelet": {"level": 42, "materials": {"Gold bar": 1, "Ruby": 1}, "mould": "bracelet mould", "xp": 80},
                "Opal bracelet": {"level": 18, "materials": {"Gold bar": 1, "Opal": 1}, "mould": "bracelet mould", "xp": 45},
                "Jade bracelet": {"level": 16, "materials": {"Gold bar": 1, "Jade": 1}, "mould": "bracelet mould", "xp": 48},
                "Topaz bracelet": {"level": 19, "materials": {"Gold bar": 1, "Topaz": 1}, "mould": "bracelet mould", "xp": 50},
            }
            
            # Filter by crafting level
            available_jewelry = {name: data for name, data in jewelry_options.items() 
                               if data["level"] <= crafting_level}
            
            if not available_jewelry:
                # Fallback to basic sapphire ring setup
                return [
                    {"name": "ring mould", "quantity": 1},
                    {"name": "Sapphire", "quantity": 13},
                    {"name": "Gold bar", "quantity": 13}
                ]
            
            # Get current GE prices for profit analysis
            all_materials = set()
            for item_data in available_jewelry.values():
                all_materials.update(item_data["materials"].keys())
            
            all_products = list(available_jewelry.keys())
            price_items = list(all_materials) + all_products
            prices = self.get_current_ge_prices(price_items)
            
            # Calculate profit for each item
            item_profits = {}
            for item_name, item_data in available_jewelry.items():
                try:
                    sell_price = prices.get(item_name, 0)
                    material_cost = 0
                    for material, qty in item_data["materials"].items():
                        material_price = prices.get(material, 0)
                        material_cost += material_price * qty
                    
                    profit = sell_price - material_cost
                    profit_per_xp = profit / item_data["xp"] if item_data["xp"] > 0 else 0
                    
                    item_profits[item_name] = {
                        "profit": profit,
                        "profit_per_xp": profit_per_xp,
                        "materials": item_data["materials"],
                        "mould": item_data["mould"]
                    }
                    
                except Exception as e:
                    logging.warning(f"[{self.id}] Error calculating profit for {item_name}: {e}")
                    continue
            
            if not item_profits:
                # Fallback to basic sapphire ring setup
                return [
                    {"name": "ring mould", "quantity": 1},
                    {"name": "Sapphire", "quantity": 13},
                    {"name": "Gold bar", "quantity": 13}
                ]
            
            # Sort by profit (most profitable first)
            sorted_items = sorted(item_profits.items(), 
                                key=lambda x: x[1]["profit"], reverse=True)
            
            # Choose the best item
            best_item_name, best_item_data = sorted_items[0]
            logging.info(f"[{self.id}] Selected optimal materials for {best_item_name} "
                        f"(Profit: {best_item_data['profit']:.2f}gp)")
            
            # Build required items list
            required_items = []
            
            # Add mould
            mould_name = best_item_data["mould"]
            required_items.append({"name": mould_name, "quantity": 1})
            
            # Add materials (13 of each for optimal crafting)
            for material, qty in best_item_data["materials"].items():
                required_items.append({"name": material, "quantity": 13})
            
            logging.info(f"[{self.id}] Optimal materials needed: {required_items}")
            return required_items
            
        except Exception as e:
            logging.error(f"[{self.id}] Error determining optimal materials: {e}")
            # Ultimate fallback
            return [
                {"name": "ring mould", "quantity": 1},
                {"name": "Sapphire", "quantity": 13},
                {"name": "Gold bar", "quantity": 13}
            ]
    
    def get_ge_items_for_crafting_level(self) -> tuple[list, str]:
        """Get GE items to buy based on current crafting level."""
        try:
            crafting_level = player.get_skill_level("crafting")
            logging.info(f"[{self.id}] Current crafting level: {crafting_level}")
            
            if crafting_level < 5:
                # Level 1-4: Buy leather, needle, thread for leather crafting
                leather_needed = self.calculate_leather_needed_for_ge()
                logging.info(f"[{self.id}] Setting up GE for leather crafting (level {crafting_level}) - need {leather_needed} leather")
                return [
                    {"name": "Needle", "quantity": 1, "bumps": 0, "set_price": 500},
                    {"name": "Thread", "quantity": 10, "bumps": 10, "set_price": 0},
                    {"name": "Leather", "quantity": leather_needed, "bumps": 5, "set_price": 0}
                ], "Leather gloves"
                
            elif crafting_level < 20:
                # Level 5-19: Buy gold bars and ring mould for gold ring crafting
                rings_needed = self.calculate_gold_rings_needed_for_level_20()
                logging.info(f"[{self.id}] Setting up GE for gold ring crafting (level {crafting_level}) - need {rings_needed} gold bars")
                return [
                    {"name": "Gold bar", "quantity": rings_needed, "bumps": 5, "set_price": 0},
                    {"name": "ring mould", "quantity": 1, "bumps": 0, "set_price": 1000}
                ], "Gold ring"
                
            else:
                # Level 20+: Use sophisticated profit analysis
                return self.get_optimal_crafting_items()
                
        except Exception as e:
            logging.error(f"[{self.id}] Error getting GE items for crafting level: {e}")
            return [], ""
    
    def get_optimal_crafting_items(self) -> tuple[list, str]:
        """Determine the most profitable crafting items based on GE prices and crafting level."""
        try:
            crafting_level = player.get_skill_level("crafting")
            # Define all possible crafting items with their requirements
            crafting_items = {
                # Rings
                "Sapphire ring": {"level": 20, "materials": {"Gold bar": 1, "Sapphire": 1}, "xp": 40},
                "Emerald ring": {"level": 27, "materials": {"Gold bar": 1, "Emerald": 1}, "xp": 55},
                "Ruby ring": {"level": 34, "materials": {"Gold bar": 1, "Ruby": 1}, "xp": 70},
                "Diamond ring": {"level": 43, "materials": {"Gold bar": 1, "Diamond": 1}, "xp": 85},
                "Opal ring": {"level": 16, "materials": {"Gold bar": 1, "Opal": 1}, "xp": 30},
                "Jade ring": {"level": 13, "materials": {"Gold bar": 1, "Jade": 1}, "xp": 32},
                "Topaz ring": {"level": 16, "materials": {"Gold bar": 1, "Topaz": 1}, "xp": 35},
                # Necklaces
                "Sapphire necklace": {"level": 22, "materials": {"Gold bar": 1, "Sapphire": 1}, "xp": 55},
                "Emerald necklace": {"level": 29, "materials": {"Gold bar": 1, "Emerald": 1}, "xp": 60},
                "Ruby necklace": {"level": 40, "materials": {"Gold bar": 1, "Ruby": 1}, "xp": 75},
                "Diamond necklace": {"level": 56, "materials": {"Gold bar": 1, "Diamond": 1}, "xp": 90},
                # Bracelets
                "Gold bracelet": {"level": 7, "materials": {"Gold bar": 1}, "xp": 11},
                "Sapphire bracelet": {"level": 23, "materials": {"Gold bar": 1, "Sapphire": 1}, "xp": 60},
                "Emerald bracelet": {"level": 30, "materials": {"Gold bar": 1, "Emerald": 1}, "xp": 65},
                "Ruby bracelet": {"level": 42, "materials": {"Gold bar": 1, "Ruby": 1}, "xp": 80},
                "Opal bracelet": {"level": 18, "materials": {"Gold bar": 1, "Opal": 1}, "xp": 45},
                "Jade bracelet": {"level": 16, "materials": {"Gold bar": 1, "Jade": 1}, "xp": 48},
                "Topaz bracelet": {"level": 19, "materials": {"Gold bar": 1, "Topaz": 1}, "xp": 50},
            }
            
            # Filter items by crafting level
            available_items = {name: data for name, data in crafting_items.items() 
                             if data["level"] <= crafting_level}
            
            if not available_items:
                logging.warning(f"[{self.id}] No craftable items available at level {crafting_level}")
                return [], ""
            
            # Get all required materials for price checking
            all_materials = set()
            for item_data in available_items.values():
                all_materials.update(item_data["materials"].keys())
            
            # Add all possible products for price checking
            all_products = list(available_items.keys())
            
            # Get current GE prices for both materials and products
            price_items = list(all_materials) + all_products
            prices = self.get_current_ge_prices(price_items)
            
            # Calculate profit for each item
            item_profits = {}
            for item_name, item_data in available_items.items():
                try:
                    # Get sell price (with bumps)
                    sell_price = prices.get(item_name, 0)
                    
                    # Calculate material costs
                    material_cost = 0
                    for material, qty in item_data["materials"].items():
                        material_price = prices.get(material, 0)
                        material_cost += material_price * qty
                    
                    # Calculate profit per item
                    profit = sell_price - material_cost
                    
                    item_profits[item_name] = {
                        "profit": profit,
                        "sell_price": sell_price,
                        "material_cost": material_cost,
                        "materials": item_data["materials"]
                    }
                    
                    logging.info(f"[{self.id}] {item_name}: Profit={profit:.0f}gp, "
                               f"Sell={sell_price:.0f}gp, Cost={material_cost:.0f}gp")
                    
                except Exception as e:
                    logging.warning(f"[{self.id}] Error calculating profit for {item_name}: {e}")
                    continue
            
            if not item_profits:
                logging.error(f"[{self.id}] No profitable items found")
                return [], ""
            
            # Sort by profit
            sorted_items = sorted(item_profits.items(), 
                                key=lambda x: x[1]["profit"], reverse=True)
            
            # Choose the best item
            best_item_name, best_item_data = sorted_items[0]
            logging.info(f"[{self.id}] Selected best item: {best_item_name} "
                        f"(Profit: {best_item_data['profit']:.2f}gp)")
            
            # Return materials needed for the best item
            items_to_buy = []
            for material, qty in best_item_data["materials"].items():
                items_to_buy.append({
                    "name": material,
                    "quantity": -1,  # Buy as much as possible
                    "bumps": 5,
                    "set_price": 0
                })
            
            return items_to_buy, best_item_name
            
        except Exception as e:
            logging.error(f"[{self.id}] Error in optimal crafting analysis: {e}")
            # Fallback to basic sapphire ring setup
            return [
                {"name": "Gold bar", "quantity": -1, "bumps": 5, "set_price": 0},
                {"name": "Sapphire", "quantity": -1, "bumps": 5, "set_price": 0}
            ], "Sapphire ring"
    
    def calculate_buy_items_from_budget(self, items_to_buy: list, budget: int) -> list:
        """Calculate what to buy based on actual available budget after selling."""
        try:
            # Get current GE prices for all items
            item_names = [item["name"] for item in items_to_buy]
            prices = self.get_current_ge_prices(item_names)
            
            # Calculate material costs for items with quantity -1
            material_costs = {}
            for item in items_to_buy:
                if item["quantity"] == -1:
                    material_price = self.calculate_buy_price(prices.get(item["name"], 0), bumps=item["bumps"])
                    material_costs[item["name"]] = material_price
            
            if not material_costs:
                logging.warning(f"[{self.id}] No items with quantity -1 found")
                return items_to_buy
            
            # Calculate total cost per "set" of all items
            total_cost_per_set = sum(material_costs.values())
            
            if total_cost_per_set <= 0:
                logging.warning(f"[{self.id}] No valid material costs found")
                return items_to_buy
            
            # Calculate how many complete sets we can afford
            max_affordable = int(budget / total_cost_per_set)
            
            if max_affordable <= 0:
                logging.warning(f"[{self.id}] Cannot afford any items with budget {budget}")
                return items_to_buy
            
            # Create buy list with calculated quantities
            calculated_items = []
            for item in items_to_buy:
                if item["quantity"] == -1:
                    # Calculate quantity based on budget
                    calculated_qty = max_affordable
                    calculated_items.append({
                        "name": item["name"],
                        "quantity": calculated_qty,
                        "bumps": item["bumps"],
                        "set_price": item["set_price"]
                    })
                    logging.info(f"[{self.id}] Will buy {calculated_qty} {item['name']}")
                else:
                    # Keep original quantity for items with specific amounts
                    calculated_items.append(item)
            
            logging.info(f"[{self.id}] Can afford {max_affordable} items with budget {budget}")
            
            return calculated_items
            
        except Exception as e:
            logging.error(f"[{self.id}] Error calculating buy items from budget: {e}")
            return items_to_buy
    
    def get_fallback_crafting_items(self) -> tuple:
        """Fallback crafting items when optimal analysis fails."""
        try:
            # Get bank state and deposit inventory
            bank_coins, existing_items = self.deposit_inventory_and_count_bank()
                
            # Get current GE prices
            prices = self.get_current_ge_prices(["Gold ring", "Sapphire ring", "Sapphire", "Gold bar"])

            # Calculate sell proceeds with bumps
            gold_rings = existing_items.get("Gold ring", 0)
            sapphire_rings = existing_items.get("Sapphire ring", 0)
            gold_ring_sell_price = self.calculate_sell_price(prices["Gold ring"], bumps=5)
            sapphire_ring_sell_price = self.calculate_sell_price(prices["Sapphire ring"], bumps=5)
            sell_proceeds = (gold_rings * gold_ring_sell_price) + (sapphire_rings * sapphire_ring_sell_price)

            budget = self.calculate_spending_budget(bank_coins, sell_proceeds)

            # Calculate buy prices with bumps
            sapphire_buy_price = self.calculate_buy_price(prices["Sapphire"], bumps=5)
            gold_bar_buy_price = self.calculate_buy_price(prices["Gold bar"], bumps=5)

            # Calculate equal purchase quantities
            sapphire_qty, gold_bar_qty = self.calculate_equal_purchase_quantities(
                budget, sapphire_buy_price, gold_bar_buy_price
            )
                
            # Items to sell
            items_to_sell = []
            if gold_rings > 0:
                items_to_sell.append({"name": "Gold ring", "quantity": -1, "bumps": 5, "set_price": 0})
            if sapphire_rings > 0:
                items_to_sell.append({"name": "Sapphire ring", "quantity": -1, "bumps": 5, "set_price": 0})

            # Items to buy
            items_to_buy = [
                {"name": "Sapphire", "quantity": sapphire_qty, "bumps": 5, "set_price": 0},
                {"name": "Gold bar", "quantity": gold_bar_qty, "bumps": 5, "set_price": 0}
            ]
                
            # Add mould if needed
            if not bank.has_item("ring mould") and not inventory.has_item("ring mould"):
                items_to_buy.append({"name": "ring mould", "quantity": 1, "bumps": 0, "set_price": 1000})
                
            return items_to_buy, items_to_sell
                
        except Exception as e:
            logging.error(f"[{self.id}] Error in fallback crafting: {e}")
            return [], []
    
    def determine_optimal_crafting_item(self, crafting_level: int) -> str:
        """Determine the most profitable item to craft based on current level and GE prices."""
        try:
            # Define all possible crafting items with their requirements
            crafting_items = {
                "Gold ring": {"level": 5, "materials": {"Gold bar": 1}, "xp": 15},
                "Gold necklace": {"level": 6, "materials": {"Gold bar": 1}, "xp": 20},
                "Sapphire ring": {"level": 20, "materials": {"Gold bar": 1, "Sapphire": 1}, "xp": 40},
                "Sapphire necklace": {"level": 22, "materials": {"Gold bar": 1, "Sapphire": 1}, "xp": 55},
                "Emerald ring": {"level": 27, "materials": {"Gold bar": 1, "Emerald": 1}, "xp": 55},
                "Emerald necklace": {"level": 29, "materials": {"Gold bar": 1, "Emerald": 1}, "xp": 60},
                "Ruby necklace": {"level": 40, "materials": {"Gold bar": 1, "Ruby": 1}, "xp": 75},
            }
            
            # Filter items by crafting level
            available_items = {name: data for name, data in crafting_items.items() 
                             if data["level"] <= crafting_level}
            
            if not available_items:
                logging.warning(f"[{self.id}] No craftable items available at level {crafting_level}")
                return "Gold ring"  # Default fallback
            
            # Get all required materials for price checking
            all_materials = set()
            for item_data in available_items.values():
                all_materials.update(item_data["materials"].keys())
            
            # Add all possible products for selling
            all_products = list(available_items.keys())
            
            # Get current GE prices
            price_items = list(all_materials) + all_products
            prices = self.get_current_ge_prices(price_items)
            
            # Calculate profit for each item
            item_profits = {}
            for item_name, item_data in available_items.items():
                try:
                    # Get sell price (with bumps)
                    sell_price = prices.get(item_name, 0)
                    
                    # Calculate material costs
                    material_cost = 0
                    for material, qty in item_data["materials"].items():
                        material_price = prices.get(material, 0)
                        material_cost += material_price * qty
                    
                    # Calculate profit per item
                    profit = sell_price - material_cost
                    profit_per_xp = profit / item_data["xp"] if item_data["xp"] > 0 else 0
                    
                    item_profits[item_name] = {
                        "profit": profit,
                        "profit_per_xp": profit_per_xp,
                        "sell_price": sell_price,
                        "material_cost": material_cost,
                        "xp": item_data["xp"],
                        "level": item_data["level"],
                        "materials": item_data["materials"]
                    }
                    
                    logging.info(f"[{self.id}] {item_name}: Profit={profit:.0f}gp, Profit/XP={profit_per_xp:.2f}gp, "
                               f"Sell={sell_price:.0f}gp, Cost={material_cost:.0f}gp")
                    
                except Exception as e:
                    logging.warning(f"[{self.id}] Error calculating profit for {item_name}: {e}")
                    continue
            
            if not item_profits:
                logging.error(f"[{self.id}] No profitable items found")
                return "Gold ring"  # Default fallback
            
            # Sort by profit per XP (most efficient first)
            sorted_items = sorted(item_profits.items(), 
                                key=lambda x: x[1]["profit"], reverse=True)
            
            # Choose the best item
            best_item_name, best_item_data = sorted_items[0]
            logging.info(f"[{self.id}] Selected best item for crafting: {best_item_name} "
                        f"(Profit/XP: {best_item_data['profit_per_xp']:.2f}gp)")
            
            return best_item_name
            
        except Exception as e:
            logging.error(f"[{self.id}] Error determining optimal crafting item: {e}")
            return "Gold ring"  # Default fallback
    
    def get_crafted_jewelry_to_sell(self) -> list:
        """Check bank and inventory for crafted jewelry items and return in sell format."""
        try:
            items_to_sell = []
            
            # Define jewelry items to check for
            jewelry_items = [
                "Gold ring",
                "Gold bracelet",
                "Sapphire ring", 
                "Sapphire necklace",
                "Sapphire bracelet",
                "Emerald ring",
                "Emerald necklace",
                "Emerald bracelet",
                "Ruby ring",
                "Ruby necklace",
                "Ruby bracelet",
                "Diamond ring",
                "Diamond necklace",
                "Opal ring",
                "Opal bracelet",
                "Jade ring",
                "Jade bracelet",
                "Topaz ring",
                "Topaz bracelet"
            ]
            
            # Check each item in both bank and inventory
            for item in jewelry_items:
                bank_count = bank.get_item_count(item)
                inventory_count = inventory.inv_count(item)
                total_count = bank_count + inventory_count
                
                if total_count > 0:
                    items_to_sell.append({
                        "name": item,
                        "quantity": -1,  # Sell all
                        "bumps": 5,
                        "set_price": 0
                    })
                    logging.info(f"[{self.id}] Found {total_count} {item} (bank: {bank_count}, inventory: {inventory_count}) - will sell all")
            
            logging.info(f"[{self.id}] Total jewelry items to sell: {len(items_to_sell)}")
            return items_to_sell
            
        except Exception as e:
            logging.error(f"[{self.id}] Error checking bank and inventory for jewelry: {e}")
            return []
    
    def setup_optimal_jewelry_crafting(self, crafting_level: int) -> dict:
        """Analyze available materials and return optimal jewelry setup data."""
        try:
            # Check what materials we have in the bank
            bank_materials = self.check_bank_materials()
            logging.info(f"[{self.id}] Bank materials available: {bank_materials}")
            
            # Define jewelry options (no gold jewelry)
            jewelry_options = {
                # Rings
                "Sapphire ring": {"level": 20, "materials": {"Gold bar": 1, "Sapphire": 1}, "mould": "ring mould", "xp": 40},
                "Emerald ring": {"level": 27, "materials": {"Gold bar": 1, "Emerald": 1}, "mould": "ring mould", "xp": 55},
                "Ruby ring": {"level": 34, "materials": {"Gold bar": 1, "Ruby": 1}, "mould": "ring mould", "xp": 70},
                "Diamond ring": {"level": 43, "materials": {"Gold bar": 1, "Diamond": 1}, "mould": "ring mould", "xp": 85},
                "Opal ring": {"level": 16, "materials": {"Gold bar": 1, "Opal": 1}, "mould": "ring mould", "xp": 30},
                "Jade ring": {"level": 13, "materials": {"Gold bar": 1, "Jade": 1}, "mould": "ring mould", "xp": 32},
                "Topaz ring": {"level": 16, "materials": {"Gold bar": 1, "Topaz": 1}, "mould": "ring mould", "xp": 35},
                # Necklaces
                "Sapphire necklace": {"level": 22, "materials": {"Gold bar": 1, "Sapphire": 1}, "mould": "necklace mould", "xp": 55},
                "Emerald necklace": {"level": 29, "materials": {"Gold bar": 1, "Emerald": 1}, "mould": "necklace mould", "xp": 60},
                "Ruby necklace": {"level": 40, "materials": {"Gold bar": 1, "Ruby": 1}, "mould": "necklace mould", "xp": 75},
                "Diamond necklace": {"level": 56, "materials": {"Gold bar": 1, "Diamond": 1}, "mould": "necklace mould", "xp": 90},
                # Bracelets
                "Sapphire bracelet": {"level": 23, "materials": {"Gold bar": 1, "Sapphire": 1}, "mould": "bracelet mould", "xp": 60},
                "Emerald bracelet": {"level": 30, "materials": {"Gold bar": 1, "Emerald": 1}, "mould": "bracelet mould", "xp": 65},
                "Ruby bracelet": {"level": 42, "materials": {"Gold bar": 1, "Ruby": 1}, "mould": "bracelet mould", "xp": 80},
                "Opal bracelet": {"level": 18, "materials": {"Gold bar": 1, "Opal": 1}, "mould": "bracelet mould", "xp": 45},
                "Jade bracelet": {"level": 16, "materials": {"Gold bar": 1, "Jade": 1}, "mould": "bracelet mould", "xp": 48},
                "Topaz bracelet": {"level": 19, "materials": {"Gold bar": 1, "Topaz": 1}, "mould": "bracelet mould", "xp": 50},
            }
            
            # Filter by crafting level and available materials
            available_jewelry = {}
            for item_name, item_data in jewelry_options.items():
                if item_data["level"] <= crafting_level:
                    # Check if we have all required materials
                    has_materials = True
                    for material, qty in item_data["materials"].items():
                        if bank_materials.get(material, 0) < qty:
                            has_materials = False
                            break
                    
                    if has_materials:
                        available_jewelry[item_name] = item_data
                        logging.info(f"[{self.id}] Can craft {item_name} - have all materials")
                    else:
                        logging.info(f"[{self.id}] Cannot craft {item_name} - missing materials")
            
            if not available_jewelry:
                # No materials available for jewelry crafting
                logging.warning(f"[{self.id}] No materials available for jewelry crafting")
                
                # Determine optimal materials needed based on crafting level and profit analysis
                optimal_materials = self.get_optimal_materials_for_missing_items(crafting_level)
                
                return {
                    "action": "missing_items",
                    "required_items": optimal_materials
                }
            
            # Use optimal analysis to select the best jewelry
            optimal_item = self.determine_optimal_crafting_item_from_available(available_jewelry, crafting_level)
            logging.info(f"[{self.id}] Selected optimal jewelry: {optimal_item}")
            
            # Set up bank plan for the selected jewelry
            if optimal_item in jewelry_options:
                item_data = jewelry_options[optimal_item]
                required_items = []
                
                # Add mould
                mould_name = item_data["mould"]
                required_items.append({"name": mould_name, "quantity": 1})
                
                # Add materials - set to lowest available quantity (max 13)
                min_quantity = 13
                for material, qty in item_data["materials"].items():
                    available_qty = bank_materials.get(material, 0)
                    if available_qty > 0:
                        min_quantity = min(min_quantity, available_qty)
                
                for material, qty in item_data["materials"].items():
                    required_items.append({"name": material, "quantity": min_quantity})
                
                logging.info(f"[{self.id}] Bank plan configured for {optimal_item}: {required_items}")
                
                return {
                    "action": "bank_setup",
                    "required_items": required_items,
                    "selected_item": optimal_item
                }
            else:
                logging.warning(f"[{self.id}] Unknown optimal item {optimal_item}, using fallback")
                
        except Exception as e:
            logging.error(f"[{self.id}] Error setting up optimal jewelry crafting: {e}")

