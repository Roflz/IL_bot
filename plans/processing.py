#!/usr/bin/env python3
"""
Processing Plan
===============

A plan for processing items. Currently supports chocolate bars.
Structured to be easily expanded for other items in the future.

Return Status Codes:
- 0: SUCCESS - Processing completed successfully
- 1: BANK_SETUP - Setting up bank for processing
- 2: ERROR - An error occurred
"""

import logging
from typing import Dict, List, Any
from pathlib import Path

# Add the parent directory to the path for imports
import sys

from actions import wait_until

sys.path.insert(0, str(Path(__file__).parent.parent))

from actions import player, bank, inventory
from actions.travel import travel_to_bank
from helpers import setup_camera_optimal
from helpers import set_phase_with_camera
from .base import Plan
from .utilities.bank_plan_simple import BankPlanSimple
from .utilities.ge import GePlan


class ProcessingPlan(Plan):
    """Processing plan for items like chocolate bars."""
    
    id = "PROCESSING"
    label = "Processing"
    description = """Processes items like chocolate bars by combining ingredients. Currently supports chocolate bar creation and can be expanded for other processing tasks.

Starting Area: Any bank
Required Items: Processing ingredients (varies by item type)"""
    
    # Return status codes
    SUCCESS = 0
    BANK_SETUP = 1
    ERROR = 2
    
    def __init__(self, item_type: str = "chocolate_bar", quantity: int = -1):
        """
        Initialize the processing plan.
        
        Args:
            item_type: Type of item to process (default: "chocolate_bar")
            quantity: Quantity of items to process (default: 28)
        """
        self.state = {"phase": "BANK_SETUP"}
        self.next = self.state["phase"]
        self.loop_interval_ms = 600
        
        # Configuration
        self.item_type = item_type
        self.quantity = quantity
        
        # Get required items based on item type
        required_items = self._get_required_items()
        
        # Bank plan for withdrawing items
        self.bank_plan = BankPlanSimple(
            bank_area=None,  # Use closest bank
            required_items=required_items,
            deposit_all=True
        )
        
        # GE plan for buying/selling items
        self.ge_plan = None
        
        # State tracking
        self.error_message = None
        
        # Set up camera immediately during initialization
        try:
            setup_camera_optimal()
        except Exception as e:
            logging.warning(f"[{self.id}] Could not setup camera: {e}")
        
        logging.info(f"[{self.id}] Processing plan initialized")
        logging.info(f"[{self.id}] Item type: {self.item_type}, Quantity: {self.quantity}")
    
    def _get_required_items(self) -> List[Dict[str, Any]]:
        """
        Get the required items for processing based on item type.
        This method can be extended to support other item types.
        
        Returns:
            List of required items with quantities
        """
        if self.item_type == "chocolate_bar":
            return [
                {"name": "Knife", "quantity": 1},
                {"name": "Chocolate bar", "quantity": self.quantity}
            ]
        else:
            # Default/unknown item type
            logging.warning(f"[{self.id}] Unknown item type: {self.item_type}, using default")
            return []
    
    def set_phase(self, phase: str, camera_setup: bool = True):
        """Set the current phase."""
        return set_phase_with_camera(self, phase, camera_setup)
    
    def loop(self, ui) -> int:
        """Main loop method following standard plan protocol."""
        phase = self.state.get("phase", "BANK_SETUP")
        logged_in = player.logged_in()
        if not logged_in:
            player.login()
            return self.loop_interval_ms

        match(phase):
            case "BANK_SETUP":
                return self._handle_bank_setup(ui)

            case "GE":
                return self._handle_ge(ui)

            case "PROCESS":
                logging.info(f"[{self.id}] Processing items...")
                return self._handle_process_items(ui)

            case "DONE":
                logging.info(f"[{self.id}] Processing complete!")
                return

        logging.warning(f"[{self.id}] Unknown phase: {phase}")
        return self.loop_interval_ms
    
    def _handle_bank_setup(self, ui) -> int:
        """Handle bank setup for processing."""
        logging.info(f"[{self.id}] Setting up bank for processing...")
        
        # Run the bank plan
        bank_status = self.bank_plan.loop(ui)
        
        if bank_status == BankPlanSimple.SUCCESS:
            logging.info(f"[{self.id}] Bank setup completed successfully")
            self.bank_plan.reset()
            self.set_phase("PROCESS")
            return
        
        elif bank_status == BankPlanSimple.MISSING_ITEMS:
            missing_items = self.bank_plan.get_missing_items()
            logging.info(f"[{self.id}] Missing items in bank: {missing_items}, will try to buy from GE")
            self.set_phase("GE")
            return
        
        elif bank_status == BankPlanSimple.ERROR:
            error_msg = self.bank_plan.get_error_message()
            logging.error(f"[{self.id}] Bank setup error: {error_msg}")
            self.error_message = f"Bank setup failed: {error_msg}"
            return
        
        else:
            # Still working on bank setup (TRAVELING, BANKING, etc.)
            return

    def _handle_ge(self, ui) -> int:
        """Handle GE phase for buying missing items and selling chocolate dust."""
        if self.ge_plan is None:
            # Ensure we're at a bank and deposit inventory
            if not travel_to_bank():
                return self.loop_interval_ms
            if not bank.is_open():
                bank.open_bank()
                return self.loop_interval_ms
            if not inventory.is_empty():
                bank.deposit_inventory()
                if not wait_until(inventory.is_empty, max_wait_ms=3000):
                    return self.loop_interval_ms
            
            # Prepare items to sell: all Chocolate dust
            items_to_sell = []
            if bank.has_item("Chocolate dust") or inventory.has_item("Chocolate dust"):
                items_to_sell.append({
                    "name": "Chocolate dust",
                    "quantity": -1,  # Sell all
                    "bumps": 5,
                    "set_price": 0
                })
            
            # Prepare items to buy
            items_to_buy = []
            
            # Check if we need a knife
            if not bank.has_item("Knife") and not inventory.has_item("Knife"):
                items_to_buy.append({
                    "name": "Knife",
                    "quantity": 1,
                    "bumps": 0,
                    "set_price": 500
                })

            items_to_buy.append({
                "name": "Chocolate bar",
                "quantity": 2000,
                "bumps": 5,
                "set_price": 0
            })
            
            # Create GE plan with both sell and buy items
            self.ge_plan = GePlan(items_to_buy=items_to_buy, items_to_sell=items_to_sell)
            logging.info(f"[{self.id}] Created GE plan - Buy: {[item['name'] for item in items_to_buy]}, Sell: {[item['name'] for item in items_to_sell]}")
        
        # Execute GE plan
        ge_status = self.ge_plan.loop(ui)
        
        if ge_status == GePlan.SUCCESS:
            logging.info(f"[{self.id}] GE phase completed successfully!")
            # Reset bank plan and go back to banking
            self.bank_plan.reset()
            self.ge_plan = None
            self.set_phase("BANK_SETUP")
            return self.loop_interval_ms
        
        elif ge_status == GePlan.ERROR:
            error_msg = self.ge_plan.get_error_message()
            logging.error(f"[{self.id}] GE plan failed: {error_msg}")
            self.set_phase("ERROR_STATE")
            self.error_message = f"GE plan failed: {error_msg}"
            return
        
        elif ge_status == GePlan.INSUFFICIENT_FUNDS:
            error_msg = self.ge_plan.get_error_message()
            logging.error(f"[{self.id}] Insufficient funds for GE purchases: {error_msg}")
            self.set_phase("ERROR_STATE")
            self.error_message = f"Insufficient funds: {error_msg}"
            return
        
        else:
            # Still working on GE (TRAVELING, SELLING, BUYING, WAITING)
            logging.debug(f"[{self.id}] GE phase in progress... Status: {ge_status}")
            return
    
    def _handle_process_items(self, ui) -> int:
        if bank.is_open():
            bank.close_bank()
            wait_until(bank.is_closed, max_wait_ms=1000)
            return
        if inventory.has_items(["Knife", "Chocolate bar"]) and not inventory.has_item("Chocolate dust"):
            inventory.use_item_on_item("Knife", "Chocolate bar")
            if wait_until(lambda: inventory.has_item("Chocolate dust"), max_wait_ms=3000):
                wait_until(lambda: not inventory.has_item("Chocolate bar"), max_wait_ms=60000)
                self.set_phase("BANK_SETUP")
                return



    
    def get_error_message(self) -> str:
        """Get the current error message."""
        return self.error_message or "Unknown error"

