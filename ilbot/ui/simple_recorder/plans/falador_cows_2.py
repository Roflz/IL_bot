#!/usr/bin/env python3
"""
Falador Cows Plan 2
==================

This plan uses the new bank plan utility to set up the character
with the same inventory and equipment configuration as the original
falador_cows.py plan, but using the simplified bank plan system.

It demonstrates how to integrate the bank plan into existing plans.
"""

import time
import logging
from pathlib import Path

# Add the parent directory to the path for imports
import sys

from ..actions import inventory, player
from ..actions.player import logged_in

sys.path.insert(0, str(Path(__file__).parent.parent))

from .base import Plan
from .utilities.bank_plan import BankPlan
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
            from ..helpers.camera import setup_camera_optimal
            setup_camera_optimal()
        except Exception as e:
            logging.warning(f"[{self.id}] Could not setup camera: {e}")
        
        # Create bank plan with the same configuration as original falador_cows
        self.bank_plan = BankPlan(
            bank_area=None,  # Use closest bank instead of specific bank
            food_item="Trout",
            food_quantity=5,  # Same as original plan
            equipment_config={
                "weapon_tiers": [
                    {"name": "Bronze scimitar", "attack_req": 1},
                    {"name": "Iron scimitar", "attack_req": 1},
                    {"name": "Steel scimitar", "attack_req": 5},
                    {"name": "Mithril scimitar", "attack_req": 20},
                    {"name": "Adamant scimitar", "attack_req": 30},
                    {"name": "Rune scimitar", "attack_req": 40}
                ],
                "armor_tiers": {
                    "helmet": [
                        {"name": "Bronze full helm", "defence_req": 1},
                        {"name": "Iron full helm", "defence_req": 1},
                        {"name": "Steel full helm", "defence_req": 5},
                        {"name": "Mithril full helm", "defence_req": 20},
                        {"name": "Adamant full helm", "defence_req": 30},
                        {"name": "Rune full helm", "defence_req": 40}
                    ],
                    "body": [
                        {"name": "Bronze platebody", "defence_req": 1},
                        {"name": "Iron platebody", "defence_req": 1},
                        {"name": "Steel platebody", "defence_req": 5},
                        {"name": "Mithril platebody", "defence_req": 20},
                        {"name": "Adamant platebody", "defence_req": 30},
                        {"name": "Rune platebody", "defence_req": 40}
                    ],
                    "legs": [
                        {"name": "Bronze platelegs", "defence_req": 1},
                        {"name": "Iron platelegs", "defence_req": 1},
                        {"name": "Steel platelegs", "defence_req": 5},
                        {"name": "Mithril platelegs", "defence_req": 20},
                        {"name": "Adamant platelegs", "defence_req": 30},
                        {"name": "Rune platelegs", "defence_req": 40}
                    ],
                    "shield": [
                        {"name": "Bronze kiteshield", "defence_req": 1},
                        {"name": "Iron kiteshield", "defence_req": 1},
                        {"name": "Steel kiteshield", "defence_req": 5},
                        {"name": "Mithril kiteshield", "defence_req": 20},
                        {"name": "Adamant kiteshield", "defence_req": 30},
                        {"name": "Rune kiteshield", "defence_req": 40}
                    ]
                },
                "jewelry_tiers": {
                    "amulet": [
                        {"name": "Amulet of strength", "defence_req": 1}
                    ]
                }
            },
            inventory_config={
                "required_items": [],  # No specific required items for cows
                "optional_items": [],
                "deposit_all": True
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
            "Mithril scimitar": {"quantity": 1, "bumps": 0, "set_price": 2000},
            "Adamant scimitar": {"quantity": 1, "bumps": 0, "set_price": 5000},
            "Rune scimitar": {"quantity": 1, "bumps": 0, "set_price": 15000},
            
            # Armor strategies by tier
            "Bronze full helm": {"quantity": 1, "bumps": 0, "set_price": 1000},
            "Iron full helm": {"quantity": 1, "bumps": 0, "set_price": 1000},
            "Steel full helm": {"quantity": 1, "bumps": 0, "set_price": 1000},
            "Mithril full helm": {"quantity": 1, "bumps": 0, "set_price": 2000},
            "Adamant full helm": {"quantity": 1, "bumps": 0, "set_price": 5000},
            "Rune full helm": {"quantity": 1, "bumps": 0, "set_price": 15000},
            
            "Bronze platebody": {"quantity": 1, "bumps": 0, "set_price": 1000},
            "Iron platebody": {"quantity": 1, "bumps": 0, "set_price": 1000},
            "Steel platebody": {"quantity": 1, "bumps": 0, "set_price": 2000},
            "Mithril platebody": {"quantity": 1, "bumps": 0, "set_price": 4000},
            "Adamant platebody": {"quantity": 1, "bumps": 0, "set_price": 10000},
            "Rune platebody": {"quantity": 1, "bumps": 0, "set_price": 30000},
            
            "Bronze platelegs": {"quantity": 1, "bumps": 0, "set_price": 1000},
            "Iron platelegs": {"quantity": 1, "bumps": 0, "set_price": 1000},
            "Steel platelegs": {"quantity": 1, "bumps": 0, "set_price": 1000},
            "Mithril platelegs": {"quantity": 1, "bumps": 0, "set_price": 3000},
            "Adamant platelegs": {"quantity": 1, "bumps": 0, "set_price": 7500},
            "Rune platelegs": {"quantity": 1, "bumps": 0, "set_price": 22500},
            
            "Bronze kiteshield": {"quantity": 1, "bumps": 0, "set_price": 1000},
            "Iron kiteshield": {"quantity": 1, "bumps": 0, "set_price": 1000},
            "Steel kiteshield": {"quantity": 1, "bumps": 0, "set_price": 1000},
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
        
        logging.info(f"[{self.id}] Plan initialized")
        logging.info(f"[{self.id}] Login credentials: {'Provided' if self.username and self.password else 'Not provided'}")
        logging.info(f"[{self.id}] Using bank plan for character setup")
        logging.info(f"[{self.id}] Food: Trout x5")
        logging.info(f"[{self.id}] Target loot: Cowhide")
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

            case "COWS":
                return self._handle_cows(ui)

        logging.warning(f"[{self.id}] Unknown phase: {phase}")
        return self.loop_interval_ms

    
    def _handle_bank(self, ui) -> int:
        """Handle banking phase - delegate all banking logic to bank plan."""
        bank_status = self.bank_plan.loop(ui)
        
        if bank_status == BankPlan.SUCCESS:
            logging.info(f"[{self.id}] Banking completed successfully!")
            # Reset attack plan so it can start fresh when we go to COWS
            if self.attack_plan is not None:
                logging.info(f"[{self.id}] Resetting attack plan for fresh start...")
                self.attack_plan.reset()
            self.set_phase("COWS")
            return self.loop_interval_ms
        
        elif bank_status == BankPlan.MISSING_ITEMS:
            error_msg = self.bank_plan.get_error_message()
            logging.warning(f"[{self.id}] Banking failed - missing items: {error_msg}")
            logging.warning(f"[{self.id}] You may need to buy items from GE or check your bank")
            self.set_phase("MISSING_ITEMS")
            return self.loop_interval_ms
        
        elif bank_status == BankPlan.ERROR:
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
            error_msg = self.bank_plan.get_error_message()
            logging.info(f"[{self.id}] Creating GE plan for missing items: {error_msg}")
            
            # Extract missing items from the error message
            # Format: "Missing required items: ['Item1', 'Item2', ...]"
            missing_items = []
            if "Missing required items:" in error_msg:
                items_str = error_msg.split("Missing required items: ")[1]
                # Remove brackets and quotes, split by comma
                items_str = items_str.strip("[]'\"")
                if items_str:
                    missing_items = [item.strip(" '\"") for item in items_str.split(",")]
            
            # Also try to parse as a Python list representation
            if not missing_items and "[" in error_msg and "]" in error_msg:
                try:
                    import ast
                    items_str = error_msg.split("Missing required items: ")[1]
                    missing_items = ast.literal_eval(items_str)
                except:
                    pass
            
            if not missing_items:
                logging.error(f"[{self.id}] Could not parse missing items from error message")

                return self.loop_interval_ms
            
            # Create GE plan with missing items using configured strategy
            items_to_buy = []
            for item_name in missing_items:
                # Use specific strategy for this item, or default if not found
                strategy = self.ge_strategy.get(item_name, self.ge_strategy["default"])
                items_to_buy.append({
                    "name": item_name,
                    "quantity": strategy["quantity"],
                    "bumps": strategy["bumps"],
                    "set_price": strategy["set_price"]
                })
            
            self.ge_plan = create_ge_plan(items_to_buy)
            logging.info(f"[{self.id}] Created GE plan to buy: {missing_items}")
        
        # Use the GE plan to buy missing items
        ge_status = self.ge_plan.loop(ui)
        
        if ge_status == GePlan.SUCCESS:
            logging.info(f"[{self.id}] Successfully purchased all missing items!")
            # Reset bank plan and try banking again
            self.bank_plan.reset()
            self.ge_plan = None  # Clear GE plan
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
        time.sleep(10)
        return self.loop_interval_ms
