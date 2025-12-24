#!/usr/bin/env python3
"""
Splash Plan
===========

This plan handles splashing activities with banking and Grand Exchange phases.
"""

import logging
import random
from pathlib import Path
import sys

from actions import inventory, player, bank, combat
from actions import can_continue
from actions.tab import open_tab
from actions import wait_until
from actions.travel import go_to, in_area
from helpers.keyboard import press_spacebar
from helpers.utils import sleep_exponential

sys.path.insert(0, str(Path(__file__).parent.parent))

from plans.base import Plan
from plans.utilities.bank_plan_simple import BankPlanSimple
from plans.utilities.ge import GePlan


class SplashPlan(Plan):
    """Splash plan with banking and GE phases."""
    
    id = "SPLASH"
    label = "Splash Plan"
    
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

        self.bank_plan = BankPlanSimple(
            bank_area=None,  # Use closest bank
            required_items=[
                {"name": "Mind rune", "quantity": -1},
                {"name": "Air rune", "quantity": -1}
            ],
            deposit_all=True,
            equip_items={
                "weapon": ["Cursed goblin staff"],
                "helmet": ["Iron full helm"],
                "body": ["Iron platebody"],
                "legs": ["Iron platelegs"],
                "shield": ["Iron kiteshield"]
            }
        )
        
        # GE strategy configuration
        # TODO: Replace placeholder items with actual items to buy/sell
        self.ge_strategy = {
            # Items to buy at GE
            # Example: "ItemName": {"quantity": 50, "bumps": 5, "set_price": 0},
            
            # Items to sell at GE
            # Example: "ItemName": {"quantity": -1, "bumps": 0, "set_price": 0},  # -1 means sell all
            
            # Default strategy for any unlisted items
            "default": {"quantity": 1, "bumps": 5, "set_price": 0}
        }
        
        # State tracking
        self.ge_plan = None  # Will be created when needed
        
        # Bank phase tracking
        self.bank_equipment_updated = False
        
        # Missing items tracking
        self.missing_items = []
        
        logging.info(f"[{self.id}] Plan initialized")
        logging.info(f"[{self.id}] Login credentials: {'Provided' if self.username and self.password else 'Not provided'}")
        logging.info(f"[{self.id}] Using simplified bank plan for character setup")
    
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

            case "GE":
                return self._handle_ge(ui)

            case "SPLASH":
                return self._handle_splash(ui)

            case "DONE":
                logging.info(f"[{self.id}] Plan complete!")
                return self.loop_interval_ms

        logging.warning(f"[{self.id}] Unknown phase: {phase}")
        return self.loop_interval_ms

    
    def _handle_bank(self, ui) -> int:
        """Handle banking phase - delegate all banking logic to bank plan."""
        # Update bank plan if needed (only once per bank phase)
        if not self.bank_equipment_updated:
            # TODO: Add any bank plan updates here if needed
            self.bank_equipment_updated = True
        
        bank_status = self.bank_plan.loop(ui)
        
        if bank_status == BankPlanSimple.SUCCESS:
            logging.info(f"[{self.id}] Banking completed successfully!")
            if bank.is_open():
                bank.close_bank()
                if not wait_until(bank.is_closed, max_wait_ms=3000):
                    return self.loop_interval_ms
            # Reset equipment update flag for next bank phase
            self.bank_equipment_updated = False
            self.bank_plan.reset()
            self.set_phase("SPLASH")
            return self.loop_interval_ms
        
        elif bank_status == BankPlanSimple.MISSING_ITEMS:
            # Get missing items directly from bank plan
            self.missing_items = self.bank_plan.get_missing_items()
            logging.warning(f"[{self.id}] Banking failed - missing items: {self.missing_items}")
            logging.warning(f"[{self.id}] You may need to buy items from GE or check your bank")
            # Reset equipment update flag for next bank phase
            self.bank_equipment_updated = False
            # Still proceed to GE to try to buy missing items
            self.set_phase("GE")
            return self.loop_interval_ms
        
        elif bank_status == BankPlanSimple.ERROR:
            error_msg = self.bank_plan.get_error_message()
            logging.error(f"[{self.id}] Banking error: {error_msg}")
            return self.loop_interval_ms
        
        else:
            # Still working on banking (TRAVELING, BANKING, EQUIPPING, etc.)
            # Return the bank plan's status so it can continue working
            return bank_status
    
    def _handle_ge(self, ui) -> int:
        """Handle Grand Exchange phase - buy and/or sell items."""
        # If we don't have a GE plan yet, create one
        if self.ge_plan is None:
            logging.info(f"[{self.id}] Creating GE plan...")
            
            # Get available coins
            total_coins = self._get_total_coins()
            logging.info(f"[{self.id}] Available coins: {total_coins}")
            
            # Build items to buy
            # TODO: Replace with actual items to buy
            items_to_buy = []
            # Example:
            # items_to_buy.append({
            #     "name": "ItemName",
            #     "quantity": 50,
            #     "bumps": 5,
            #     "set_price": 0
            # })
            
            # Build items to sell
            # TODO: Replace with actual items to sell
            items_to_sell = []
            # Example:
            # items_to_sell.append({
            #     "name": "ItemName",
            #     "quantity": -1,  # -1 means sell all
            #     "bumps": 0,
            #     "set_price": 0
            # })
            
            # Also add any missing items from banking to the buy list
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
            
            if not items_to_buy and not items_to_sell:
                logging.info(f"[{self.id}] No items to buy or sell, completing plan")
                self.set_phase("DONE")
                return self.loop_interval_ms
            
            self.ge_plan = GePlan(
                items_to_buy=items_to_buy if items_to_buy else None,
                items_to_sell=items_to_sell if items_to_sell else None
            )
            logging.info(f"[{self.id}] Created GE plan")
            if items_to_buy:
                logging.info(f"[{self.id}] Items to buy: {[item['name'] for item in items_to_buy]}")
            if items_to_sell:
                logging.info(f"[{self.id}] Items to sell: {[item['name'] for item in items_to_sell]}")
        
        # Use the GE plan to buy/sell items
        ge_status = self.ge_plan.loop(ui)
        
        if ge_status == GePlan.SUCCESS:
            logging.info(f"[{self.id}] Successfully completed GE operations!")
            # Clear missing items if we bought them
            self.missing_items = []
            self.ge_plan = None  # Clear GE plan
            self.set_phase("DONE")
            return self.loop_interval_ms
        
        elif ge_status == GePlan.ERROR:
            error_msg = self.ge_plan.get_error_message()
            logging.error(f"[{self.id}] GE plan failed: {error_msg}")
            return self.loop_interval_ms
        
        elif ge_status == GePlan.INSUFFICIENT_FUNDS:
            error_msg = self.ge_plan.get_error_message()
            logging.error(f"[{self.id}] Insufficient funds for GE purchases: {error_msg}")
            # Still mark as done since we can't proceed
            self.set_phase("DONE")
            return self.loop_interval_ms
        
        else:
            # Still working on GE operations (TRAVELING, CHECKING_COINS, BUYING, SELLING, WAITING)
            logging.debug(f"[{self.id}] GE plan in progress... Status: {ge_status}")
            return ge_status

    def _handle_splash(self, ui):
        if not inventory.has_items(["Mind rune", "Air rune"]):
            if not wait_until(lambda: inventory.has_items(["Mind rune", "Air rune"]), max_wait_ms=1200):
                self.set_phase("DONE")
                return
        if not player.is_in_combat():
            if can_continue():
                press_spacebar()
                return
            if not in_area((3106, 3121, 3519, 3520)):
                go_to((3106, 3121, 3519, 3520))
                return
            if inventory.has_items(["Mind rune", "Air rune"]):
                combat.attack_closest("Skeleton")
                return 3000
            return
        elif player.is_in_combat():
            if can_continue():
                press_spacebar()
                return sleep_exponential(0.3, 3, 1.0)
            # sleep_exponential(30, 1800, 2.0)
            
            # Randomly choose between 4 different action paths
            if inventory.has_items(["Mind rune", "Air rune"]):
                action_choice = random.randint(1, 4000)

                match action_choice:
                    case 1:
                        open_tab("SKILLS")
                        pass
                    case 2:
                        open_tab("INVENTORY")
                        pass
                    case 3:
                        open_tab("COMBAT")
                        pass
                    case 4:
                        open_tab("SPELLBOOK")
                        pass
                    case 5:
                        open_tab("RANDOM")
                        pass

        return self.loop_interval_ms
    
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

