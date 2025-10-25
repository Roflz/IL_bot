#!/usr/bin/env python3
"""
Woodcutting Plan 2
==================

This plan uses the new bank plan and GE plan utilities to handle woodcutting.
It demonstrates how to integrate the utility plans for a complete skilling activity.

Features:
- Uses BankPlan for character setup and banking
- Uses GePlan for buying missing axes and selling logs
- Uses custom tree cutting logic
- Automatically selects best axe for your level
- Handles equipment and inventory management
- Smart log selling: Automatically sells logs to afford better axes when:
  * Current axe is not the best available for your level
  * Have 100+ logs in bank
  * Don't have enough coins to buy the better axe
"""

import time
import logging
from pathlib import Path

# Add the parent directory to the path for imports
import sys

from ..helpers.camera import setup_camera_optimal
from ..helpers.phase_utils import set_phase_with_camera

sys.path.insert(0, str(Path(__file__).parent.parent))

from .base import Plan
from .utilities.bank_plan_simple import BankPlanSimple
from .utilities.ge import GePlan, create_ge_plan
from ..actions import objects, player, chat, inventory, equipment, bank, ge
from ..actions.timing import wait_until
from ..helpers.utils import exponential_number
from ..actions.travel import in_area, go_to, travel_to_bank


class Woodcutting2Plan(Plan):
    """Woodcutting plan using the new bank plan and GE plan utilities."""
    
    id = "WOODCUTTING_2"
    label = "Woodcutting 2 - Using Bank Plan"
    
    def __init__(self):
        self.state = {"phase": "BANK"}
        self.next = self.state["phase"]
        self.loop_interval_ms = 600
        
        # Set up camera immediately during initialization
        try:
            setup_camera_optimal()
        except Exception as e:
            logging.warning(f"[{self.id}] Could not setup camera: {e}")
        
        # Configuration
        self.tree_area = "VARROCK_WEST_TREES"
        self.bank_area = "VARROCK_WEST"
        self.tree_type = "Tree"
        self.log_name = "Logs"
        
        # Axe options in order of preference (best to worst)
        # Format: (axe_name, woodcutting_level, attack_level, defence_level)
        self.axe_options = [
            ("Dragon axe", 61, 60, 1),
            ("Rune axe", 41, 40, 1),
            ("Adamant axe", 31, 30, 1),
            ("Mithril axe", 21, 20, 1),
            ("Black axe", 11, 10, 1),
            ("Steel axe", 6, 5, 1),
            ("Iron axe", 1, 1, 1),
            ("Bronze axe", 1, 1, 1)
        ]
        
        # Create simple bank plan - just deposit inventory, no equipment for now
        # We'll handle equipment setup separately if needed
        self.bank_plan = BankPlanSimple(
            bank_area=self.bank_area,
            required_items=[],  # No required items for now
            deposit_all=True,
            equip_items={}  # No equipment for now, we'll add this later
        )
        
        # GE strategy configuration - specific strategies for each item
        self.ge_strategy = {
            # Axe strategies by tier
            "Bronze axe": {"quantity": 1, "bumps": 0, "set_price": 1000},
            "Iron axe": {"quantity": 1, "bumps": 0, "set_price": 1000},
            "Steel axe": {"quantity": 1, "bumps": 0, "set_price": 1000},
            "Black axe": {"quantity": 1, "bumps": 0, "set_price": 2000},
            "Mithril axe": {"quantity": 1, "bumps": 0, "set_price": 2000},
            "Adamant axe": {"quantity": 1, "bumps": 0, "set_price": 5000},
            "Rune axe": {"quantity": 1, "bumps": 0, "set_price": 10000},
            "Dragon axe": {"quantity": 1, "bumps": 0, "set_price": 50000},
            
            # Log selling strategy
            "Logs": {"quantity": -1, "bumps": 5, "set_price": 0},  # Sell at market price
            
            # Default strategy for any unlisted items
            "default": {"quantity": 1, "bumps": 0, "set_price": 1000}
        }
        
        # State tracking
        self.ge_plan = None  # Will be created when needed
        self.bank_equipment_updated = False
        
        logging.info(f"[{self.id}] Plan initialized")
        logging.info(f"[{self.id}] Using bank plan for character setup")
        logging.info(f"[{self.id}] Tree area: {self.tree_area}")
        logging.info(f"[{self.id}] Bank area: {self.bank_area}")
        logging.info(f"[{self.id}] Tree type: {self.tree_type}")
        logging.info(f"[{self.id}] Collecting: {self.log_name}")
        logging.info(f"[{self.id}] GE strategy: {self.ge_strategy}")
    
    def set_phase(self, phase: str, camera_setup: bool = True):
        """Set the current phase."""
        return set_phase_with_camera(self, phase, camera_setup)
    
    def loop(self, ui) -> int:
        """Main loop method."""
        phase = self.state.get("phase", "BANK")
        logged_in = player.logged_in()
        if not logged_in:
            logging.info(f"pplayer not logged in")
            player.login()
            return self.loop_interval_ms

        match(phase):
            case "BANK":
                return self._handle_bank(ui)

            case "MISSING_ITEMS":
                return self._handle_missing_items(ui)

            case "WOODCUTTING":
                return self._handle_woodcutting()

        logging.warning(f"[{self.id}] Unknown phase: {phase}")
        return self.loop_interval_ms

    
    def _handle_bank(self, ui) -> int:
        """Handle banking phase - determine best axe and configure bank plan."""
        # Update bank plan with best axe for current skill level (only once per bank phase)
        if not self.bank_equipment_updated:
            if not travel_to_bank():
                return self.loop_interval_ms
            if not bank.is_open():
                bank.open_bank()
                return self.loop_interval_ms
            # First, determine best axe based on woodcutting level
            woodcutting_level = player.get_skill_level("woodcutting") or 1
            best_axe = self._determine_best_axe_for_level(woodcutting_level)
            
            logging.info(f"[{self.id}] Best axe for woodcutting level {woodcutting_level}: {best_axe}")
            
            # Check if we have this axe in bank, inventory, or equipment
            has_axe = (bank.has_item(best_axe) or 
                      inventory.has_item(best_axe) or 
                      equipment.has_equipped(best_axe))
            
            if not has_axe:
                # Try to get the axe through GE purchase or log selling
                axe_result = self._try_to_get_axe(best_axe, woodcutting_level)
                if axe_result == "GE_PLAN_CREATED":
                    return self.loop_interval_ms
                elif axe_result == "FALLBACK_AXE":
                    best_axe = self._find_best_available_axe(woodcutting_level)
                    logging.info(f"[{self.id}] Using best available axe: {best_axe}")
            
            # Now configure bank plan based on attack level and axe we'll use
            attack_level = player.get_skill_level("attack") or 1
            self._configure_bank_plan_for_axe(best_axe, attack_level)
            
            self.bank_equipment_updated = True
        
        bank_status = self.bank_plan.loop(ui)
        
        if bank_status == BankPlanSimple.SUCCESS:
            logging.info(f"[{self.id}] Banking completed successfully!")
            
            # Reset bank plan so it can start fresh when we go back to banking
            if self.bank_plan is not None:
                logging.info(f"[{self.id}] Resetting bank plan for fresh start...")
                self.bank_plan.reset()
            # Reset equipment update flag for next bank phase
            self.bank_equipment_updated = False
            self.set_phase("WOODCUTTING")
            return self.loop_interval_ms
        
        elif bank_status == BankPlanSimple.MISSING_ITEMS:
            # This should not happen with our new logic, but handle it anyway
            logging.info(f"[{self.id}] Missing items detected")
            missing_items = self.bank_plan.get_missing_items()
            logging.info(f"[{self.id}] Missing items: {missing_items}")
            
            # Create GE plan to buy missing items
            if missing_items:
                items_to_buy = []
                for item in missing_items:
                    if isinstance(item, dict):
                        item_name = item.get("name")
                        quantity = item.get("quantity", 1)
                        strategy = self.ge_strategy.get(item_name, self.ge_strategy["default"])
                        items_to_buy.append({
                            "name": item_name,
                            "quantity": quantity,
                            "bumps": strategy.get("bumps", 0),
                            "set_price": strategy.get("set_price", 0)
                        })
                
                if items_to_buy:
                    self.ge_plan = create_ge_plan(items_to_buy, [])
                    logging.info(f"[{self.id}] Created GE plan to buy: {items_to_buy}")
            
            # Reset bank plan so it can start fresh when we go back to banking
            if self.bank_plan is not None:
                logging.info(f"[{self.id}] Resetting bank plan for fresh start...")
                self.bank_plan.reset()
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
        # GE plan should have been created in _handle_bank
        if self.ge_plan is None:
            logging.error(f"[{self.id}] No GE plan created, returning to bank")
            self.bank_plan.reset()
            self.bank_equipment_updated = False
            self.set_phase("BANK")
            return self.loop_interval_ms
        
        logging.info(f"[{self.id}] Running GE plan to buy axe...")
        
        # Use the GE plan to buy missing items
        ge_status = self.ge_plan.loop(ui)
        
        if ge_status == GePlan.SUCCESS:
            logging.info(f"[{self.id}] Successfully purchased axe from GE!")
            # Clear GE plan
            self.ge_plan = None
            # Reset equipment update flag so we reconfigure
            self.bank_equipment_updated = False
            # Go back to bank to equip the axe
            self.bank_plan.reset()
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
    
    def _handle_woodcutting(self) -> int:
        """Handle woodcutting phase."""
        if player.get_run_energy() > 2000 and not player.is_run_on():
            player.toggle_run()
        if bank.is_open():
            bank.close_bank()
            return self.loop_interval_ms
        if inventory.is_full():
            logging.info(f"[{self.id}] Inventory full, returning to bank")
            # Reset bank plan so it can run through the banking process again
            self.bank_plan.reset()
            self.set_phase("BANK")
            return self.loop_interval_ms
        
        # Handle chat dialogues
        if chat.can_continue():
            chat.continue_dialogue()
            return self.loop_interval_ms
        
        # Check if we're already chopping
        if player.get_player_animation() == "CHOPPING":
            return self.loop_interval_ms

        # Check if we're in the tree area
        if not in_area(self.tree_area):
            logging.info(f"[{self.id}] Traveling to {self.tree_area}...")
            go_to(self.tree_area)
            return self.loop_interval_ms

        # Look for trees to cut
        logging.info(f"[{self.id}] Looking for a tree to cut")
        tree = objects.click_object_in_area(self.tree_type, self.tree_area, "Chop down")
        if tree:
            wait_until(lambda: player.get_player_animation() == "CHOPPING", max_wait_ms=5000)
        
        return self.loop_interval_ms
    
    def _handle_error(self) -> int:
        """Handle error state."""
        logging.error(f"[{self.id}] Plan is in error state")
        logging.error(f"[{self.id}] Check logs above for details")
        
        # Wait and let user see the error
        import time
        wait_time = exponential_number(8, 12, 1.5)
        time.sleep(wait_time)
        return self.loop_interval_ms
    
    def _determine_best_axe_for_level(self, woodcutting_level: int) -> str:
        """Determine the best axe for woodcutting level (no attack level check)."""
        try:
            # Find best axe based on woodcutting level
            for axe_info in self.axe_options:
                axe_name, woodcutting_req, attack_req, defence_req = axe_info
                if woodcutting_level >= woodcutting_req:
                    return axe_name
            
            # Fallback to worst axe
            return ""
            
        except Exception as e:
            logging.warning(f"[{self.id}] Error determining best axe for level: {e}")
            return ""
    
    def _find_best_available_axe(self, woodcutting_level: int) -> str:
        """Find the best available axe we actually have."""
        try:
            # Check each axe from best to worst that we can use
            for axe_info in self.axe_options:
                axe_name, woodcutting_req, attack_req, defence_req = axe_info
                if woodcutting_level >= woodcutting_req:
                    # Check if we have this axe
                    if (bank.has_item(axe_name) or 
                        inventory.has_item(axe_name) or 
                        equipment.has_equipped(axe_name)):
                        logging.info(f"[{self.id}] Found available axe: {axe_name}")
                        return axe_name
            
            # Fallback to worst axe
            logging.warning(f"[{self.id}] No available axe found")
            return ""
            
        except Exception as e:
            logging.warning(f"[{self.id}] Error finding available axe: {e}")
            return ""
    
    def _configure_bank_plan_for_axe(self, axe_name: str, attack_level: int) -> None:
        """Configure bank plan based on whether we can equip the axe."""
        try:
            # Find attack requirement for this axe
            axe_attack_req = 0
            for axe_info in self.axe_options:
                if axe_info[0] == axe_name:
                    axe_attack_req = axe_info[2]  # attack_req is at index 2
                    break
            
            if attack_level >= axe_attack_req:
                # Can equip, add to equip_items
                logging.info(f"[{self.id}] Can equip {axe_name} (attack: {attack_level} >= {axe_attack_req})")
                self.bank_plan.equip_items = {"weapon": [axe_name]}
            else:
                # Cannot equip, add to required_items
                logging.info(f"[{self.id}] Cannot equip {axe_name} (attack: {attack_level} < {axe_attack_req}), adding to required_items")
                self.bank_plan.required_items = [{"name": axe_name, "quantity": 1}]
                self.bank_plan.equip_items = {}
            
        except Exception as e:
            logging.warning(f"[{self.id}] Error configuring bank plan for axe: {e}")
    
    def _get_total_coins(self) -> int:
        """Get total coins from bank and inventory."""
        try:
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
    
    def _get_total_logs(self) -> int:
        """Get total logs from bank and inventory."""
        try:
            total_logs = 0
            
            # Get logs from bank
            if bank.has_item(self.log_name):
                total_logs += bank.get_item_count(self.log_name)
            
            # Get logs from inventory
            if inventory.has_item(self.log_name):
                total_logs += inventory.inv_count(self.log_name)
            
            return total_logs
        except Exception as e:
            logging.warning(f"[{self.id}] Error getting log count: {e}")
            return 0
    
    def _try_to_get_axe(self, axe_name: str, woodcutting_level: int) -> str:
        """
        Try to get an axe through GE purchase or log selling.
        
        Returns:
            "GE_PLAN_CREATED" - GE plan was created, phase should change to MISSING_ITEMS
            "FALLBACK_AXE" - Cannot afford axe, should use best available axe
        """
        logging.info(f"[{self.id}] Don't have {axe_name}, checking if we can afford it...")
        
        # Get the set price for this axe
        strategy = self.ge_strategy.get(axe_name, self.ge_strategy["default"])
        set_price = strategy.get("set_price", 0)
        
        # Check total coins
        total_coins = self._get_total_coins()
        
        if set_price > 0 and total_coins >= set_price:
            logging.info(f"[{self.id}] Have enough coins ({total_coins} >= {set_price}) to buy {axe_name}")
            # Create GE plan to buy the axe
            items_to_buy = [{
                "name": axe_name,
                "quantity": 1,
                "bumps": strategy.get("bumps", 0),
                "set_price": set_price
            }]
            self.ge_plan = create_ge_plan(items_to_buy, [])
            logging.info(f"[{self.id}] Created GE plan to buy: {axe_name}")
            self.set_phase("MISSING_ITEMS")
            return "GE_PLAN_CREATED"
        
        # Check if we can sell logs to afford the axe
        logging.info(f"[{self.id}] Don't have enough coins ({total_coins} < {set_price}), checking if we can sell logs...")
        
        # Get total logs count
        total_logs = self._get_total_logs()
        logging.info(f"[{self.id}] Total logs available: {total_logs}")
        
        if total_logs > 0:
            # Get actual GE prices for logs and axe
            items_to_price = [self.log_name, axe_name]
            ge_prices = ge.get_current_ge_prices(items_to_price)
            
            # Calculate log sell price with bumps (selling at reduced price)
            log_strategy = self.ge_strategy.get(self.log_name, self.ge_strategy["default"])
            log_bumps = log_strategy.get("bumps", 5)  # Default to 5 bumps for selling
            
            base_log_price = ge_prices.get(self.log_name, 50)  # Fallback to 50 if not found
            sell_price_per_log = ge.calculate_sell_price(base_log_price, log_bumps)
            estimated_total_value = total_logs * sell_price_per_log
            
            logging.info(f"[{self.id}] Log pricing: base={base_log_price}, bumps={log_bumps}, sell_price={sell_price_per_log}")
            logging.info(f"[{self.id}] Total log value: {total_logs} logs × {sell_price_per_log} = {estimated_total_value} coins")
            
            if estimated_total_value >= set_price:
                logging.info(f"[{self.id}] Can afford {axe_name} by selling logs! Creating GE plan...")
                
                # Create GE plan to sell logs and buy axe
                items_to_sell = [{
                    "name": self.log_name,
                    "quantity": -1,  # Sell exact number of logs we have
                    "bumps": log_bumps,
                    "set_price": 0  # Market price
                }]
                
                items_to_buy = [{
                    "name": axe_name,
                    "quantity": 1,
                    "bumps": strategy.get("bumps", 0),
                    "set_price": set_price
                }]
                
                self.ge_plan = create_ge_plan(items_to_buy, items_to_sell)
                logging.info(f"[{self.id}] Created GE plan to sell {total_logs} logs and buy: {axe_name}")
                self.set_phase("MISSING_ITEMS")
                return "GE_PLAN_CREATED"
            else:
                logging.info(f"[{self.id}] Not enough logs to afford {axe_name} (estimated {estimated_total_value} < {set_price})")
        else:
            logging.info(f"[{self.id}] No logs available to sell")
        
        logging.warning(f"[{self.id}] Cannot afford {axe_name}, falling back to available axes")
        return "FALLBACK_AXE"
    


# Helper function for easy setup
def create_woodcutting_plan(tree_area: str = "VARROCK_WEST_TREES",
                           bank_area: str = "VARROCK_WEST",
                           tree_type: str = "Tree") -> Woodcutting2Plan:
    """
    Create a woodcutting plan with custom configuration.
    
    Args:
        tree_area: Area name where trees are located
        bank_area: Bank area to use for banking
        tree_type: Type of tree to cut
    
    Returns:
        Configured Woodcutting2Plan instance
    """
    plan = Woodcutting2Plan()
    plan.tree_area = tree_area
    plan.bank_area = bank_area
    plan.tree_type = tree_type
    return plan
