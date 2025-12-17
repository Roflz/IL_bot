#!/usr/bin/env python3
"""
Falador Cows Plan 2
==================

This plan uses the simplified bank plan utility to set up the character
with combat equipment and food for fighting cows in Falador.

It demonstrates how to use BankPlanSimple for straightforward character setup.
"""

import logging
from pathlib import Path

# Add the parent directory to the path for imports
import sys

from actions import inventory, player, bank
from actions import wait_until

sys.path.insert(0, str(Path(__file__).parent.parent))

from .base import Plan
from .utilities.bank_plan_simple import BankPlanSimple
from .utilities.ge import GePlan, create_ge_plan
from .utilities.attack_npcs import AttackNpcsPlan, create_cow_attack_plan


class FaladorCows2Plan(Plan):
    """Falador cows plan using the new bank plan utility."""
    
    id = "FALADOR_COWS_2"
    label = "Falador Cows 2 - Using Bank Plan"
    
    def __init__(self, username: str = None, password: str = None):
        self.state = {"phase": "BANK"}
        self.next = self.state["phase"]
        self.loop_interval_ms = 600
        
        # Login credentials
        self.username = username
        self.password = password
        
        # Set up camera immediately during initialization
        try:
            from helpers import setup_camera_optimal
            setup_camera_optimal()
        except Exception as e:
            logging.warning(f"[{self.id}] Could not setup camera: {e}")
        
        # Create simple bank plan for cow fighting
        self.bank_plan = BankPlanSimple(
            bank_area=None,  # Use closest bank
            required_items=[{"name": "Trout", "quantity": 5}],  # Food for combat
            deposit_all=True,
            equip_items={
                "weapon": ["Rune scimitar", "Adamant scimitar", "Mithril scimitar", "Steel scimitar", "Iron scimitar", "Bronze scimitar"],
                "helmet": ["Rune full helm", "Adamant full helm", "Mithril full helm", "Steel full helm", "Iron full helm", "Bronze full helm"],
                "body": ["Rune platebody", "Adamant platebody", "Mithril platebody", "Steel platebody", "Iron platebody", "Bronze platebody"],
                "legs": ["Rune platelegs", "Adamant platelegs", "Mithril platelegs", "Steel platelegs", "Iron platelegs", "Bronze platelegs"],
                "shield": ["Rune kiteshield", "Adamant kiteshield", "Mithril kiteshield", "Steel kiteshield", "Iron kiteshield", "Bronze kiteshield"],
                "amulet": ["Amulet of glory", "Amulet of power", "Amulet of strength"]
            }
        )
        
        # GE strategy configuration - specific strategies for each item
        self.ge_strategy = {
            # Food strategies
            "Trout": {"quantity": 50, "bumps": 15, "set_price": 0},
            "Salmon": {"quantity": 50, "bumps": 5, "set_price": 0},
            "Lobster": {"quantity": 50, "bumps": 5, "set_price": 0},
            
            # Weapon strategies by tier
            "Bronze scimitar": {"quantity": 1, "bumps": 0, "set_price": 1000},
            "Iron scimitar": {"quantity": 1, "bumps": 0, "set_price": 1000},
            "Steel scimitar": {"quantity": 1, "bumps": 0, "set_price": 1000},
            "Black scimitar": {"quantity": 1, "bumps": 0, "set_price": 10000},
            "Mithril scimitar": {"quantity": 1, "bumps": 0, "set_price": 2000},
            "Adamant scimitar": {"quantity": 1, "bumps": 0, "set_price": 5000},
            "Rune scimitar": {"quantity": 1, "bumps": 0, "set_price": 15000},
            
            # Armor strategies by tier
            "Bronze full helm": {"quantity": 1, "bumps": 0, "set_price": 1000},
            "Iron full helm": {"quantity": 1, "bumps": 0, "set_price": 1000},
            "Steel full helm": {"quantity": 1, "bumps": 0, "set_price": 1000},
            "Black full helm": {"quantity": 1, "bumps": 0, "set_price": 3000},
            "Mithril full helm": {"quantity": 1, "bumps": 0, "set_price": 2000},
            "Adamant full helm": {"quantity": 1, "bumps": 0, "set_price": 5000},
            "Rune full helm": {"quantity": 1, "bumps": 0, "set_price": 15000},
            
            "Bronze platebody": {"quantity": 1, "bumps": 0, "set_price": 1000},
            "Iron platebody": {"quantity": 1, "bumps": 0, "set_price": 1000},
            "Steel platebody": {"quantity": 1, "bumps": 0, "set_price": 2000},
            "Black platebody": {"quantity": 1, "bumps": 0, "set_price": 10000},
            "Mithril platebody": {"quantity": 1, "bumps": 0, "set_price": 4000},
            "Adamant platebody": {"quantity": 1, "bumps": 0, "set_price": 10000},
            "Rune platebody": {"quantity": 1, "bumps": 0, "set_price": 30000},
            
            "Bronze platelegs": {"quantity": 1, "bumps": 0, "set_price": 1000},
            "Iron platelegs": {"quantity": 1, "bumps": 0, "set_price": 1000},
            "Steel platelegs": {"quantity": 1, "bumps": 0, "set_price": 1000},
            "Black platelegs": {"quantity": 1, "bumps": 0, "set_price": 5000},
            "Mithril platelegs": {"quantity": 1, "bumps": 0, "set_price": 3000},
            "Adamant platelegs": {"quantity": 1, "bumps": 0, "set_price": 7500},
            "Rune platelegs": {"quantity": 1, "bumps": 0, "set_price": 22500},
            
            "Bronze kiteshield": {"quantity": 1, "bumps": 0, "set_price": 1000},
            "Iron kiteshield": {"quantity": 1, "bumps": 0, "set_price": 1000},
            "Steel kiteshield": {"quantity": 1, "bumps": 0, "set_price": 1000},
            "Black kiteshield": {"quantity": 1, "bumps": 0, "set_price": 5000},
            "Mithril kiteshield": {"quantity": 1, "bumps": 0, "set_price": 2000},
            "Adamant kiteshield": {"quantity": 1, "bumps": 0, "set_price": 5000},
            "Rune kiteshield": {"quantity": 1, "bumps": 0, "set_price": 15000},
            
            # Jewelry strategies
            "Amulet of strength": {"quantity": 1, "bumps": 0, "set_price": 3000},
            "Amulet of power": {"quantity": 1, "bumps": 0, "set_price": 5000},
            "Amulet of glory": {"quantity": 1, "bumps": 0, "set_price": 10000},
            
            # Default strategy for any unlisted items
            "default": {"quantity": 1, "bumps": 5, "set_price": 0}
        }
        
        # State tracking
        self.ge_plan = None  # Will be created when needed
        self.attack_plan = None  # Will be created when needed
        
        # Bank phase tracking
        self.bank_equipment_updated = False
        
        # Missing items tracking
        self.missing_items = []
        
        logging.info(f"[{self.id}] Plan initialized")
        logging.info(f"[{self.id}] Login credentials: {'Provided' if self.username and self.password else 'Not provided'}")
        logging.info(f"[{self.id}] Using simplified bank plan for character setup")
        logging.info(f"[{self.id}] Food: Trout x5")
        logging.info(f"[{self.id}] Equipment: Best available combat gear with fallbacks")
        logging.info(f"[{self.id}] Target loot: Cowhide")
        logging.info(f"[{self.id}] GE strategy: {self.ge_strategy}")
    
    def set_phase(self, phase: str, camera_setup: bool = True):
        """Set the current phase."""
        from helpers import set_phase_with_camera
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

            case "COWS":
                return self._handle_cows(ui)

        logging.warning(f"[{self.id}] Unknown phase: {phase}")
        return self.loop_interval_ms

    
    def _handle_bank(self, ui) -> int:
        """Handle banking phase - delegate all banking logic to bank plan."""
        # Update bank plan with best equipment for current skill level (only once per bank phase)
        if not self.bank_equipment_updated:
            self._update_bank_plan_equipment()
            self.bank_equipment_updated = True
        
        bank_status = self.bank_plan.loop(ui)
        
        if bank_status == BankPlanSimple.SUCCESS:
            logging.info(f"[{self.id}] Banking completed successfully!")
            if bank.is_open():
                bank.close_bank()
                if not wait_until(bank.is_closed, max_wait_ms=3000):
                    return self.loop_interval_ms
            # Reset attack plan so it can start fresh when we go to COWS
            if self.attack_plan is not None:
                logging.info(f"[{self.id}] Resetting attack plan for fresh start...")
                self.attack_plan.reset()
            # Reset equipment update flag for next bank phase
            self.bank_equipment_updated = False
            self.set_phase("COWS")
            return self.loop_interval_ms
        
        elif bank_status == BankPlanSimple.MISSING_ITEMS:
            # Get missing items directly from bank plan
            self.missing_items = self.bank_plan.get_missing_items()
            logging.warning(f"[{self.id}] Banking failed - missing items: {self.missing_items}")
            logging.warning(f"[{self.id}] You may need to buy items from GE or check your bank")
            # Reset equipment update flag for next bank phase
            self.bank_equipment_updated = False
            self.set_phase("MISSING_ITEMS")
            return self.loop_interval_ms
        
        elif bank_status == BankPlanSimple.ERROR:
            error_msg = self.bank_plan.get_error_message()
            logging.error(f"[{self.id}] Banking error: {error_msg}")

            return self.loop_interval_ms
        
        else:
            # Still working on banking (TRAVELING, BANKING, EQUIPPING, etc.)
            # Return the bank plan's status so it can continue working
            return bank_status
    
    def _handle_missing_items(self, ui) -> int:
        """Handle missing items phase by using GE utility."""
        # If we don't have a GE plan yet, create one with the missing items
        if self.ge_plan is None:
            logging.info(f"[{self.id}] Creating GE plan for missing items: {self.missing_items}")
            
            # Get available coins
            total_coins = self._get_total_coins()
            logging.info(f"[{self.id}] Available coins: {total_coins}")
            
            # Create buy plan with smart equipment purchasing
            items_to_buy = []
            
            # First, add any missing required items (like food)
            for missing_item in self.missing_items:
                item_name = missing_item["name"]
                needed_quantity = missing_item["quantity"]
                strategy = self.ge_strategy.get(item_name, self.ge_strategy["default"])
                items_to_buy.append({
                    "name": item_name,
                    "quantity": strategy['quantity'],
                    "bumps": strategy["bumps"],
                    "set_price": strategy["set_price"]
                })
            
            
            if not items_to_buy:
                logging.error(f"[{self.id}] No items to buy - insufficient funds or no missing items")
                return self.loop_interval_ms
            
            self.ge_plan = create_ge_plan(items_to_buy)
            logging.info(f"[{self.id}] Created GE plan to buy: {[item['name'] for item in items_to_buy]}")
        
        # Use the GE plan to buy missing items
        ge_status = self.ge_plan.loop(ui)
        
        if ge_status == GePlan.SUCCESS:
            logging.info(f"[{self.id}] Successfully purchased all missing items!")
            # Reset bank plan and try banking again
            self.bank_plan.reset()
            self.ge_plan = None  # Clear GE plan
            # Clear missing items and reset equipment update flag for fresh bank phase
            self.missing_items = []
            self.bank_equipment_updated = False
            self.set_phase("BANK")
            return self.loop_interval_ms
        
        elif ge_status == GePlan.ERROR:
            error_msg = self.ge_plan.get_error_message()
            logging.error(f"[{self.id}] GE plan failed: {error_msg}")
            return self.loop_interval_ms
        
        elif ge_status == GePlan.INSUFFICIENT_FUNDS:
            error_msg = self.ge_plan.get_error_message()
            logging.error(f"[{self.id}] Insufficient funds for GE purchases: {error_msg}")
            return self.loop_interval_ms
        
        else:
            # Still working on buying items (TRAVELING, CHECKING_COINS, BUYING, WAITING)
            logging.info(f"[{self.id}] GE plan in progress... Status: {ge_status}")
            return ge_status
    
    def _handle_cows(self, ui) -> int:
        """Handle cow killing phase using attack_npcs utility."""
        # Check if inventory is full - if so, return to bank
        empty_slots = inventory.get_empty_slots_count()
        if empty_slots == 0:
            logging.info(f"[{self.id}] Inventory full, returning to bank")
            # Reset bank plan so it can run through the banking process again
            self.bank_plan.reset()
            self.set_phase("BANK")
            return self.loop_interval_ms
        
        # Create attack plan if we don't have one yet
        if self.attack_plan is None:
            logging.info(f"[{self.id}] Creating attack plan for cows...")
            self.attack_plan = create_cow_attack_plan()
        
        # Use the attack plan to handle cow killing
        attack_status = self.attack_plan.loop(ui)
        
        if attack_status == AttackNpcsPlan.SUCCESS:
            logging.info(f"[{self.id}] Attack session completed!")
            # Reset attack plan for next session
            self.attack_plan.reset()
            # Continue attacking (don't transition to another phase)
            return self.loop_interval_ms
        
        elif attack_status == AttackNpcsPlan.ERROR:
            error_msg = self.attack_plan.get_error_message()
            logging.error(f"[{self.id}] Attack plan failed: {error_msg}")

            return self.loop_interval_ms
        
        else:
            # Still working on attacking (TRAVELING, ATTACKING)
            logging.debug(f"[{self.id}] Attack plan in progress... Status: {attack_status}")
            return attack_status
    
    def _handle_error(self) -> int:
        """Handle error state."""
        logging.error(f"[{self.id}] Plan is in error state")
        logging.error(f"[{self.id}] Check logs above for details")
        
        # Wait and let user see the error
        from helpers.utils import sleep_exponential
        sleep_exponential(8, 12, 1.5)
        return self.loop_interval_ms
    
    def _update_bank_plan_equipment(self) -> None:
        """Update the bank plan's equipment configuration based on current skill levels."""
        try:
            from actions import get_skill_level
            
            # Get current skill levels
            attack_level = get_skill_level("attack") or 1
            defence_level = get_skill_level("defence") or 1
            
            logging.info(f"[{self.id}] Current levels - Attack: {attack_level}, Defence: {defence_level}")
            
            # Determine best equipment for current levels
            best_equipment = self._determine_best_equipment()
            
            # Create equipment lists with best equipment first, then fallbacks
            updated_equip_items = {}
            
            # Weapon: best weapon first, then fallbacks
            if best_equipment.get("weapon"):
                weapon_list = []
                # Add fallbacks that are worse than our best
                best = False
                for weapon in ["Rune scimitar", "Adamant scimitar", "Mithril scimitar", "Black scimitar", "Steel scimitar", "Iron scimitar", "Bronze scimitar"]:
                    if not best:
                        if weapon == best_equipment["weapon"]:
                            best = True
                            weapon_list.append(weapon)
                    else:
                        weapon_list.append(weapon)
                updated_equip_items["weapon"] = weapon_list
            else:
                # No weapon available for our level, use full fallback list
                updated_equip_items["weapon"] = ["Rune scimitar", "Adamant scimitar", "Mithril scimitar", "Steel scimitar", "Steel scimitar", "Iron scimitar", "Bronze scimitar"]
            
            # Armor: best armor first, then fallbacks for each slot
            for slot in ["helmet", "body", "legs", "shield"]:
                if best_equipment.get("armor", {}).get(slot):
                    armor_list = []
                    # Add fallbacks for this slot
                    fallback_armors = {
                        "helmet": ["Rune full helm", "Adamant full helm", "Mithril full helm", "Black full helm", "Steel full helm", "Iron full helm", "Bronze full helm"],
                        "body": ["Rune platebody", "Adamant platebody", "Mithril platebody", "Black platebody", "Steel platebody", "Iron platebody", "Bronze platebody"],
                        "legs": ["Rune platelegs", "Adamant platelegs", "Mithril platelegs", "Black platelegs", "Steel platelegs", "Iron platelegs", "Bronze platelegs"],
                        "shield": ["Rune kiteshield", "Adamant kiteshield", "Mithril kiteshield", "Black kiteshield", "Steel kiteshield", "Iron kiteshield", "Bronze kiteshield"]
                    }
                    best = False
                    for armor in fallback_armors[slot]:
                        if not best:
                            if armor == best_equipment["armor"][slot]:
                                best = True
                                armor_list.append(armor)
                        else:
                            armor_list.append(armor)
                    updated_equip_items[slot] = armor_list
                else:
                    # No armor available for our level, use full fallback list
                    fallback_armors = {
                        "helmet": ["Rune full helm", "Adamant full helm", "Mithril full helm", "Black full helm", "Steel full helm", "Iron full helm", "Bronze full helm"],
                        "body": ["Rune platebody", "Adamant platebody", "Mithril platebody", "Black platebody", "Steel platebody", "Iron platebody", "Bronze platebody"],
                        "legs": ["Rune platelegs", "Adamant platelegs", "Mithril platelegs", "Black platelegs", "Steel platelegs", "Iron platelegs", "Bronze platelegs"],
                        "shield": ["Rune kiteshield", "Adamant kiteshield", "Mithril kiteshield", "Black kiteshield", "Steel kiteshield", "Iron kiteshield", "Bronze kiteshield"]
                    }
                    updated_equip_items[slot] = fallback_armors[slot]
            
            # Jewelry: best jewelry first, then fallbacks
            if best_equipment.get("jewelry", {}).get("amulet"):
                amulet_list = []
                # Add fallbacks
                best = False
                for amulet in ["Amulet of glory", "Amulet of power", "Amulet of strength"]:
                    if not best:
                        if amulet == best_equipment["jewelry"]["amulet"]:
                            best = True
                            amulet_list.append(amulet)
                    else:
                        amulet_list.append(amulet)
                updated_equip_items["amulet"] = amulet_list
            else:
                # Use full fallback list
                updated_equip_items["amulet"] = ["Amulet of glory", "Amulet of power", "Amulet of strength"]
            
            # Update the bank plan's equipment configuration
            self.bank_plan.equip_items = updated_equip_items
            
            logging.info(f"[{self.id}] Updated bank plan equipment:")
            for slot, items in updated_equip_items.items():
                logging.info(f"[{self.id}]   {slot}: {items[0]} (best) + {len(items)-1} fallbacks")
            
        except Exception as e:
            logging.warning(f"[{self.id}] Error updating bank plan equipment: {e}")
            # Keep using the original equipment configuration if there's an error
    
    def _get_total_coins(self) -> int:
        """Get total coins from bank and inventory."""
        try:
            from actions import bank, inventory
            
            total_coins = 0
            
            # Get coins from bank
            if bank.has_item("Coins"):
                total_coins += bank.get_item_count("Coins")
            
            # Get coins from inventory
            if inventory.has_item("Coins"):
                total_coins += inventory.inv_count("Coins")
            
            return total_coins
        except Exception as e:
            logging.warning(f"[{self.id}] Error getting coin count: {e}")
            return 0
    
    def _determine_best_equipment(self) -> dict:
        """Determine the best equipment for our current skill level."""
        try:
            from actions import get_skill_level
            
            # Get current skill levels
            attack_level = get_skill_level("attack") or 1
            defence_level = get_skill_level("defence") or 1
            
            # Define equipment tiers with requirements
            weapon_tiers = [
                {"name": "Rune scimitar", "attack_req": 40},
                {"name": "Adamant scimitar", "attack_req": 30},
                {"name": "Mithril scimitar", "attack_req": 20},
                {"name": "Black scimitar", "attack_req": 10},
                {"name": "Steel scimitar", "attack_req": 5},
                {"name": "Iron scimitar", "attack_req": 1},
                {"name": "Bronze scimitar", "attack_req": 1}
            ]
            
            armor_tiers = {
                "helmet": [
                    {"name": "Rune full helm", "defence_req": 40},
                    {"name": "Adamant full helm", "defence_req": 30},
                    {"name": "Mithril full helm", "defence_req": 20},
                    {"name": "Black full helm", "defence_req": 10},
                    {"name": "Steel full helm", "defence_req": 5},
                    {"name": "Iron full helm", "defence_req": 1},
                    {"name": "Bronze full helm", "defence_req": 1}
                ],
                "body": [
                    {"name": "Rune platebody", "defence_req": 40},
                    {"name": "Adamant platebody", "defence_req": 30},
                    {"name": "Mithril platebody", "defence_req": 20},
                    {"name": "Black platebody", "defence_req": 10},
                    {"name": "Steel platebody", "defence_req": 5},
                    {"name": "Iron platebody", "defence_req": 1},
                    {"name": "Bronze platebody", "defence_req": 1}
                ],
                "legs": [
                    {"name": "Rune platelegs", "defence_req": 40},
                    {"name": "Adamant platelegs", "defence_req": 30},
                    {"name": "Mithril platelegs", "defence_req": 20},
                    {"name": "Black platelegs", "defence_req": 10},
                    {"name": "Steel platelegs", "defence_req": 5},
                    {"name": "Iron platelegs", "defence_req": 1},
                    {"name": "Bronze platelegs", "defence_req": 1}
                ],
                "shield": [
                    {"name": "Rune kiteshield", "defence_req": 40},
                    {"name": "Adamant kiteshield", "defence_req": 30},
                    {"name": "Mithril kiteshield", "defence_req": 20},
                    {"name": "Black kiteshield", "defence_req": 10},
                    {"name": "Steel kiteshield", "defence_req": 5},
                    {"name": "Iron kiteshield", "defence_req": 1},
                    {"name": "Bronze kiteshield", "defence_req": 1}
                ]
            }
            
            # Find best weapon
            best_weapon = None
            for weapon in weapon_tiers:
                if attack_level >= weapon["attack_req"]:
                    best_weapon = weapon["name"]
                    break
            
            # Find best armor for each slot
            best_armor = {}
            for slot, armors in armor_tiers.items():
                for armor in armors:
                    if defence_level >= armor["defence_req"]:
                        best_armor[slot] = armor["name"]
                        break
            
            # Add jewelry (no level requirements)
            best_jewelry = {
                "amulet": "Amulet of strength"  # Best amulet
            }
            
            return {
                "weapon": best_weapon,
                "armor": best_armor,
                "jewelry": best_jewelry
            }
            
        except Exception as e:
            logging.warning(f"[{self.id}] Error determining best equipment: {e}")
            return {"weapon": None, "armor": {}, "jewelry": {}}
    
    def _check_missing_equipment(self, best_equipment: dict) -> list:
        """Check which equipment we're missing."""
        try:
            from actions import bank, inventory, equipment
            
            missing = []
            
            # Check weapon
            if best_equipment.get("weapon"):
                weapon_name = best_equipment["weapon"]
                if not (equipment.has_equipped(weapon_name) or
                        inventory.has_item(weapon_name) or
                        bank.has_item(weapon_name)):
                    missing.append(weapon_name)
            
            # Check armor
            for slot, armor_name in best_equipment.get("armor", {}).items():
                if not (equipment.has_equipped(armor_name) or
                        inventory.has_item(armor_name) or
                        bank.has_item(armor_name)):
                    missing.append(armor_name)
            
            # Check jewelry
            for slot, jewelry_name in best_equipment.get("jewelry", {}).items():
                if not (equipment.has_equipped(jewelry_name) or
                        inventory.has_item(jewelry_name) or
                        bank.has_item(jewelry_name)):
                    missing.append(jewelry_name)
            
            return missing
            
        except Exception as e:
            logging.warning(f"[{self.id}] Error checking missing equipment: {e}")
            return []
    
    def _calculate_affordable_equipment(self, missing_equipment: list, available_coins: int) -> list:
        """Calculate which equipment we can afford to buy."""
        try:
            affordable = []
            remaining_coins = available_coins
            
            # Process weapons first
            weapon_items = [item for item in missing_equipment if "scimitar" in item.lower()]
            for weapon in weapon_items:
                estimated_cost = self._estimate_item_cost(weapon)
                if remaining_coins >= estimated_cost:
                    affordable.append(weapon)
                    remaining_coins -= estimated_cost
                    logging.info(f"[{self.id}] Can afford {weapon} (cost: ~{estimated_cost}, remaining: {remaining_coins})")
                else:
                    logging.info(f"[{self.id}] Cannot afford {weapon} (cost: ~{estimated_cost}, available: {remaining_coins})")
            
            # Then process armor (helmet, body, legs, shield)
            armor_items = [item for item in missing_equipment if item not in weapon_items]
            for armor in armor_items:
                estimated_cost = self._estimate_item_cost(armor)
                if remaining_coins >= estimated_cost:
                    affordable.append(armor)
                    remaining_coins -= estimated_cost
                    logging.info(f"[{self.id}] Can afford {armor} (cost: ~{estimated_cost}, remaining: {remaining_coins})")
                else:
                    logging.info(f"[{self.id}] Cannot afford {armor} (cost: ~{estimated_cost}, available: {remaining_coins})")
            
            return affordable
            
        except Exception as e:
            logging.warning(f"[{self.id}] Error calculating affordable equipment: {e}")
            return []
    
    def _estimate_item_cost(self, item_name: str) -> int:
        """Estimate the cost of an item based on its tier."""
        # Use the set_price from our GE strategy as a rough estimate
        strategy = self.ge_strategy.get(item_name, self.ge_strategy["default"])
        return strategy["set_price"] if strategy["set_price"] > 0 else 1000  # Default fallback
