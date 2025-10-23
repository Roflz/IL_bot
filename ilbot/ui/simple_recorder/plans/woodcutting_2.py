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
sys.path.insert(0, str(Path(__file__).parent.parent))

from .base import Plan
from .utilities.bank_plan import BankPlan
from .utilities.ge import GePlan, create_ge_plan
from ..actions import objects, player, chat, inventory, equipment
from ..actions.timing import wait_until
from ..actions.travel import in_area, go_to


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
            from ..helpers.camera import setup_camera_optimal
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
        
        # Create bank plan with axe equipment configuration
        self.bank_plan = BankPlan(
            bank_area=self.bank_area,
            food_item=None,  # No food for woodcutting
            food_quantity=0,
            equipment_config={
                "weapon_tiers": [],  # No weapons needed for woodcutting
                "armor_tiers": {},   # No armor needed for woodcutting
                "jewelry_tiers": {}, # No jewelry needed for woodcutting
                "tool_tiers": self.axe_options  # Axes as tools
            },
            inventory_config={
                "required_items": [],  # No specific required items
                "optional_items": [],
                "deposit_all": True
            },
            sellable_items={
                self.log_name: 50  # Check for 50+ logs to sell
            }
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
            "Logs": {"quantity": 50, "bumps": 5, "set_price": 0},  # Sell at market price
            
            # Default strategy for any unlisted items
            "default": {"quantity": 1, "bumps": 0, "set_price": 1000}
        }
        
        # State tracking
        self.ge_plan = None  # Will be created when needed
        
        logging.info(f"[{self.id}] Plan initialized")
        logging.info(f"[{self.id}] Using bank plan for character setup")
        logging.info(f"[{self.id}] Tree area: {self.tree_area}")
        logging.info(f"[{self.id}] Bank area: {self.bank_area}")
        logging.info(f"[{self.id}] Tree type: {self.tree_type}")
        logging.info(f"[{self.id}] Collecting: {self.log_name}")
        logging.info(f"[{self.id}] GE strategy: {self.ge_strategy}")
    
    def set_phase(self, phase: str, camera_setup: bool = True):
        """Set the current phase."""
        from ..helpers.phase_utils import set_phase_with_camera
        return set_phase_with_camera(self, phase, camera_setup)
    
    def loop(self, ui) -> int:
        """Main loop method."""
        phase = self.state.get("phase", "BANK")
        logging.info(f"checking login")
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
        """Handle banking phase - delegate all banking logic to bank plan."""
        bank_status = self.bank_plan.loop(ui)
        
        if bank_status == BankPlan.SUCCESS:
            logging.info(f"[{self.id}] Banking completed successfully!")
            
            # Reset bank plan so it can start fresh when we go back to banking
            if self.bank_plan is not None:
                logging.info(f"[{self.id}] Resetting bank plan for fresh start...")
                self.bank_plan.reset()
            self.set_phase("WOODCUTTING")
            return self.loop_interval_ms
        
        elif bank_status == BankPlan.ITEMS_TO_SELL:
            logging.info(f"[{self.id}] Bank found items to sell!")
            
            # Get selling information from bank plan
            sellable_items = self.bank_plan.get_sellable_items()
            target_equipment = self.bank_plan.get_target_equipment()
            
            logging.info(f"[{self.id}] Items to sell: {sellable_items}")
            logging.info(f"[{self.id}] Target equipment: {target_equipment}")
            
            # Create GE plan to sell items and buy equipment
            self._create_sell_and_buy_ge_plan(sellable_items, target_equipment)
            
            # Reset bank plan so it can start fresh when we go back to banking
            if self.bank_plan is not None:
                logging.info(f"[{self.id}] Resetting bank plan for fresh start...")
                self.bank_plan.reset()
            self.set_phase("MISSING_ITEMS")
            return self.loop_interval_ms
        
        elif bank_status == BankPlan.MISSING_ITEMS:
            error_msg = self.bank_plan.get_error_message()
            logging.warning(f"[{self.id}] Banking failed - missing items: {error_msg}")
            logging.warning(f"[{self.id}] Will try to buy best axe from GE")
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
    
    def _create_sell_and_buy_ge_plan(self, sellable_items: dict, target_equipment: str):
        """Create a GE plan to sell items and buy equipment."""
        try:
            # Create items to sell list
            items_to_sell = []
            for item_name, item_count in sellable_items.items():
                # Get selling strategy for this item
                strategy = self.ge_strategy.get(item_name, self.ge_strategy["default"])
                items_to_sell.append({
                    "name": item_name,
                    "quantity": item_count,
                    "bumps": strategy["bumps"],
                    "set_price": strategy["set_price"]
                })
            
            # Create items to buy list
            items_to_buy = []
            if target_equipment:
                # Get buying strategy for the target equipment
                strategy = self.ge_strategy.get(target_equipment, self.ge_strategy["default"])
                items_to_buy.append({
                    "name": target_equipment,
                    "quantity": 1,
                    "bumps": strategy["bumps"],
                    "set_price": strategy["set_price"]
                })
            
            # Create the GE plan
            self.ge_plan = create_ge_plan(items_to_buy, items_to_sell)
            logging.info(f"[{self.id}] Created GE plan to sell {sellable_items} and buy {target_equipment}")
            
        except Exception as e:
            logging.error(f"[{self.id}] Error creating sell and buy GE plan: {e}")
            # Don't fail the plan, just continue with woodcutting
    
    def _handle_missing_items(self, ui) -> int:
        """Handle missing items phase by using GE utility."""
        # If we don't have a GE plan yet, create one with the missing items
        if self.ge_plan is None:
            error_msg = self.bank_plan.get_error_message()
            logging.info(f"[{self.id}] Creating GE plan for missing items: {error_msg}")
            
            # Extract missing items from the error message
            missing_items = []
            if "Missing required items:" in error_msg:
                items_str = error_msg.split("Missing required items: ")[1]
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
            for item_info in missing_items:
                # Parse item info - could be "ItemName" or "ItemName|quantity"
                if "|" in item_info:
                    item_name, needed_quantity = item_info.split("|", 1)
                    needed_quantity = int(needed_quantity)
                else:
                    item_name = item_info
                    needed_quantity = 1
                
                # Use specific strategy for this item, or default if not found
                strategy = self.ge_strategy.get(item_name, self.ge_strategy["default"])
                items_to_buy.append({
                    "name": item_name,
                    "quantity": needed_quantity,  # Use the actual needed quantity
                    "bumps": strategy["bumps"],
                    "set_price": strategy["set_price"]
                })

            logging.info(f"[{self.id}] Created GE plan to buy: {missing_items}")
        else:
            logging.info(f"[{self.id}] Using existing GE plan (likely for selling logs and buying better axe)")
        
        # Use the GE plan to buy missing items
        ge_status = self.ge_plan.loop(ui)
        
        if ge_status == GePlan.SUCCESS:
            logging.info(f"[{self.id}] Successfully purchased all missing items!")
            # Reset bank plan and try banking again
            self.bank_plan.reset()
            self.ge_plan = None
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
        # Check if inventory is full - if so, return to bank
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
        time.sleep(10)
        return self.loop_interval_ms
    


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
