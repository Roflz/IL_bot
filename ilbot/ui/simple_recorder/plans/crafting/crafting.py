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
from typing import List, Dict

# Add the parent directory to the path for imports
import sys

from ...actions import player, bank, inventory, objects, chat, widgets
from ...actions.bank import deposit_inventory
from ...actions.timing import wait_until
from ...actions.travel import travel_to_bank, go_to
from ...helpers.camera import setup_camera_optimal
from ...helpers.phase_utils import set_phase_with_camera
from ...helpers.utils import sleep_exponential, exponential_number
from ...helpers.keyboard import press_spacebar
from ...helpers.widgets import widget_exists
from ...constants import JEWELRY_CRAFTING_WIDGETS

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ..base import Plan
from ..utilities.bank_plan_simple import BankPlanSimple
from ..utilities.ge import GePlan, create_ge_plan
from .methods import CraftingMethods


class CraftingPlan(Plan):
    """Crafting plan using BankPlan for banking and custom crafting logic."""
    
    id = "CRAFTING"
    label = "Crafting: Edgeville Bank"
    
    def __init__(self):
        self.state = {"phase": "BANK"}
        self.next = self.state["phase"]
        self.loop_interval_ms = 600
        
        # Initialize helper methods
        self.methods = CraftingMethods(self.id)
        
        # Set up camera immediately during initialization
        try:
            setup_camera_optimal()
        except Exception as e:
            logging.warning(f"[{self.id}] Could not setup camera: {e}")
        
        # Configuration
        self.bank_area = None
        self.crafting_items = {
            "gold ring": (28, "ring mould", "sapphire"),  # (quantity, mould, gem)
            "gold necklace": (28, "necklace mould", "emerald")
        }
        
        # Create simplified bank plan for banking
        self.bank_plan = BankPlanSimple(
            bank_area=self.bank_area,
            required_items=[],  # Will be set dynamically based on crafting level
            deposit_all=True
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
        self.bank_plan_updated = False  # Track if we've updated bank plan for this session
        
        logging.info(f"[{self.id}] Plan initialized")
        logging.info(f"[{self.id}] Using bank plan for character setup")
        logging.info(f"[{self.id}] Bank area: {self.bank_area}")
        logging.info(f"[{self.id}] Crafting items: {self.crafting_items}")
        logging.info(f"[{self.id}] GE strategy: {self.ge_strategy}")
    
    def set_phase(self, phase: str, camera_setup: bool = True):
        """Set the current phase."""
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
        # Update bank plan inventory config based on crafting level (only once per session)
        if not self.bank_plan_updated:
            if not travel_to_bank():
                return self.loop_interval_ms
            if not bank.is_open():
                bank.open_bank()
                return self.loop_interval_ms

            self._update_bank_plan_for_crafting_level()
            self.bank_plan_updated = True
        
        # Check if we're transitioning to MISSING_ITEMS (skip bank loop in this case)
        if self.state.get("phase") == "MISSING_ITEMS":
            return self.loop_interval_ms
        
        bank_status = self.bank_plan.loop(ui)
        
        if bank_status == BankPlanSimple.SUCCESS:
            logging.info(f"[{self.id}] Banking completed successfully!")
            self.bank_plan_updated = False  # Reset for next banking session
            self.set_phase("CRAFTING")
            return bank_status
        
        elif bank_status == BankPlanSimple.MISSING_ITEMS:
            missing_items = self.bank_plan.get_missing_items()
            logging.warning(f"[{self.id}] Banking failed - missing items: {missing_items}")
            logging.warning(f"[{self.id}] Will try to buy missing items from GE")
            self.bank_plan_updated = False  # Reset for next banking session
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
                leather_needed = self.methods.calculate_leather_needed_for_level_5()
                logging.info(f"[{self.id}] Setting up for leather crafting (level {crafting_level}) - need {leather_needed} leather")
                self.bank_plan.required_items = [
                    {"name": "Needle", "quantity": 1},
                    {"name": "Thread", "quantity": 10},
                    {"name": "Leather", "quantity": leather_needed}
                ]
                
            elif crafting_level < 20:
                # Level 5-19: Gold ring crafting - calculate how many rings needed
                rings_needed = self.methods.calculate_gold_rings_needed_for_level_20()
                logging.info(f"[{self.id}] Setting up for gold ring crafting (level {crafting_level}) - need {rings_needed} gold bars")
                self.bank_plan.required_items = [
                    {"name": "ring mould", "quantity": 1},
                    {"name": "Gold bar", "quantity": -1}
                ]
                
            else:
                # Level 20+: Check what materials we have and select optimal jewelry
                jewelry_result = self.methods.setup_optimal_jewelry_crafting(crafting_level)
                
                if jewelry_result["action"] == "missing_items":
                    # No materials available, switch to MISSING_ITEMS phase
                    self.bank_plan.required_items = jewelry_result["required_items"]
                    self.bank_plan_updated = True  # Mark as updated so we skip the bank loop
                    self.set_phase("MISSING_ITEMS")
                else:
                    # Materials available, set up bank plan
                    self.bank_plan.required_items = jewelry_result["required_items"]
                    logging.info(f"[{self.id}] Selected jewelry: {jewelry_result.get('selected_item', 'unknown')}")
                
        except Exception as e:
            logging.error(f"[{self.id}] Error updating bank plan for crafting level: {e}")
            # Default to basic gold ring setup
            self.bank_plan.required_items = [
                {"name": "Gold bar", "quantity": 28},
                {"name": "ring mould", "quantity": 1}
            ]
    
    def _handle_missing_items(self, ui) -> int:
        """Handle missing items phase using two-phase GE execution."""
        # Phase 1: Sell items first
        if not hasattr(self, 'ge_sell_phase_complete'):
            if self.ge_plan is None:
                if not self.bank_plan_updated:
                    if not travel_to_bank():
                        return self.loop_interval_ms
                    if not bank.is_open():
                        bank.open_bank()
                        return self.loop_interval_ms
                if not inventory.is_empty():
                    deposit_inventory()
                    if not wait_until(inventory.is_empty, max_wait_ms=3000):
                        return self.loop_interval_ms
                sell_items = self.methods.get_crafted_jewelry_to_sell()

                    
                # Create GE plan for selling only
                self.ge_plan = GePlan(items_to_buy=[], items_to_sell=sell_items)
                logging.info(f"[{self.id}] Created GE sell plan: {[item['name'] for item in sell_items]}")
            
            # Execute sell phase
            ge_status = self.ge_plan.loop(ui)
            
            if ge_status == GePlan.SUCCESS:
                logging.info(f"[{self.id}] Sell phase completed successfully!")
                self.ge_sell_phase_complete = True
                self.ge_plan = None  # Clear for buy phase
                return self.loop_interval_ms
            else:
                # Still working on selling
                logging.info(f"[{self.id}] GE sell phase in progress... Status: {ge_status}")
                return ge_status
        
        # Phase 2: Calculate budget and buy items
        else:
            if self.ge_plan is None:
                if not travel_to_bank():
                    return self.loop_interval_ms
                if not bank.is_open():
                    bank.open_bank()
                    return self.loop_interval_ms
                # Calculate actual budget after selling
                # Get current coins
                current_coins = bank.get_item_count("coins") if bank.is_open() else 0
                current_coins += inventory.inv_count("coins")

                # Calculate what to buy based on real budget
                items_to_buy, best_item_name = self.methods.get_ge_items_for_crafting_level()
                # Add mould if needed
                if "ring" in best_item_name.lower():
                    if not bank.has_item("ring mould") and not inventory.has_item("ring mould"):
                        items_to_buy.append({
                                    "name": "ring mould",
                                    "quantity": 1,
                                    "bumps": 0,
                                    "set_price": 1000
                                })
                        current_coins -= 1000
                elif "necklace" in best_item_name.lower():
                    if not bank.has_item("necklace mould") and not inventory.has_item("necklace mould"):
                        items_to_buy.append({
                                    "name": "necklace mould",
                            "quantity": 1,
                            "bumps": 0,
                            "set_price": 1000
                        })
                        current_coins -= 1000
                items_to_buy = self.methods.calculate_buy_items_from_budget(items_to_buy, current_coins)

                if items_to_buy:
                    # Create GE plan for buying only
                    self.ge_plan = GePlan(items_to_buy=items_to_buy, items_to_sell=[])
                    logging.info(f"[{self.id}] Created GE buy plan: {[item['name'] for item in items_to_buy]}")
                else:
                    logging.warning(f"[{self.id}] No items to buy with current budget")
                    # Reset and try banking again
                    self.bank_plan.reset()
                    self.bank_plan_updated = False
                    delattr(self, 'ge_sell_phase_complete')
                    delattr(self, 'crafting_analysis')
                    self.set_phase("BANK")
                    return self.loop_interval_ms
            
        # Execute buy phase`
        ge_status = self.ge_plan.loop(ui)
        
        if ge_status == GePlan.SUCCESS:
            logging.info(f"[{self.id}] Buy phase completed successfully!")
            # Reset everything and try banking again
            self.bank_plan.reset()
            self.bank_plan_updated = False
            self.ge_plan = None
            delattr(self, 'ge_sell_phase_complete')
            if hasattr(self, 'crafting_analysis'):
                delattr(self, 'crafting_analysis')
            self.set_phase("BANK")
            return self.loop_interval_ms
        else:
            # Still working on buying
            logging.info(f"[{self.id}] GE buy phase in progress... Status: {ge_status}")
            return ge_status
    
    def _handle_crafting(self) -> int:
        """Handle crafting phase."""
        if not objects.object_exists("furnace", 25) and inventory.has_item("gold bar"):
            go_to("EDGEVILLE_BANK")
            return self.loop_interval_ms

        if ((not inventory.has_unnoted_item("gold bar") or not inventory.has_unnoted_item(self.bank_plan.required_items[2].get("name")))
                and not inventory.has_unnoted_item("leather")):
            # Reset bank plan so it can run through the banking process again
            self.bank_plan.reset()
            self.bank_plan_updated = False  # Reset flag so we update bank plan again
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

        if inventory.is_empty(["gold bar", "ring mould", "necklace mould"]):
            objects.click_object_closest_by_distance_simple("Furnace", "Smelt")
            if not wait_until(lambda: widget_exists(29229056)): # Crafting interface
                return self.loop_interval_ms
            sleep_exponential(0.3, 0.8, 1.2)
            opts = widgets.get_crafting_options_rings()
            for widget in opts:
                if widget.get('itemId') == 1635:
                    widgets.click_widget(widget.get("id"))
                    break

            if not wait_until(lambda: player.get_player_animation() == "SMELTING"):
                return self.loop_interval_ms
            return self.loop_interval_ms

        # Determine what we're crafting based on bank plan
        mould_type = None
        gem_type = None
        
        # Check bank plan for mould and gem types
        for item in self.bank_plan.required_items:
            if "mould" in item["name"].lower():
                mould_type = "ring" if "ring" in item["name"].lower() else "necklace"
            elif item["name"].lower() in ["sapphire", "emerald", "ruby"]:
                gem_type = item["name"].lower()
        
        # Check if we have the required materials
        required_materials = {"gold bar": "any"}
        if mould_type:
            required_materials[f"{mould_type} mould"] = 1
        if gem_type:
            required_materials[gem_type] = "any"
        
        if inventory.has_items(required_materials):
            objects.click_object_closest_by_distance_simple("furnace", "smelt")
            if not wait_until(lambda: widget_exists(29229056)): # Crafting interface
                return self.loop_interval_ms
            
            # Get appropriate crafting options based on mould type
            sleep_exponential(0.3, 1, 1)
            
            # Use the unified craft_jewelry method
            self.craft_jewelry(mould_type, gem_type)

            if not wait_until(lambda: player.get_player_animation() == "SMELTING"):
                return self.loop_interval_ms
            return self.loop_interval_ms
        
        return self.loop_interval_ms
    
    def craft_jewelry(self, mould_type: str, gem_type: str) -> bool:
        """
        Craft jewelry using the appropriate widget based on mould and gem type.
        
        Args:
            mould_type: "ring" or "necklace"
            gem_type: "sapphire", "emerald", "ruby", etc.
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Determine the widget key based on mould and gem type
            widget_key = f"{gem_type.upper()}_{mould_type.upper()}"
            
            # Get the widget ID from constants
            widget_id = JEWELRY_CRAFTING_WIDGETS.get(widget_key)
            
            if not widget_id:
                logging.error(f"[{self.id}] Unknown jewelry combination: {mould_type} + {gem_type}")
                return False
            
            # Click the widget
            success = widgets.click_widget(widget_id)
            
            if success:
                logging.info(f"[{self.id}] Successfully clicked {widget_key} (ID: {widget_id})")
            else:
                logging.error(f"[{self.id}] Failed to click {widget_key} (ID: {widget_id})")
            
            return success
            
        except Exception as e:
            logging.error(f"[{self.id}] Error in craft_jewelry: {e}")
            return False
