#!/usr/bin/env python3
"""
Example Plan Using Bank Plan
===========================

This is an example of how to integrate the bank plan into another plan.
It shows how to use the bank plan as a utility within your main plan logic.
"""

import time
import logging
from pathlib import Path

# Add the parent directory to the path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from ..actions import player, inventory
from .base import Plan
from .bank_plan import BankPlan


class ExampleWithBankPlan(Plan):
    """Example plan that uses the bank plan for setup."""
    
    id = "EXAMPLE_WITH_BANK"
    label = "Example Plan Using Bank Setup"
    
    def __init__(self):
        self.state = {"phase": "BANK_SETUP"}
        self.next = self.state["phase"]
        self.loop_interval_ms = 600
        
        # Create bank plan instance
        self.bank_plan = BankPlan(
            bank_area="FALADOR_BANK",
            food_item="Trout",
            food_quantity=20,
            inventory_config={
                "required_items": ["Rope", "Tinderbox"],
                "optional_items": ["Coins"],
                "deposit_all": True
            }
        )
        
        logging.info(f"[{self.id}] Example plan initialized with bank setup")
    
    def set_phase(self, phase: str, camera_setup: bool = True):
        """Set the current phase."""
        from ..helpers.phase_utils import set_phase_with_camera
        return set_phase_with_camera(self, phase, camera_setup)
    
    def loop(self, ui) -> int:
        """Main loop method."""
        phase = self.state.get("phase", "BANK_SETUP")
        
        try:
            match(phase):
                case "BANK_SETUP":
                    return self._handle_bank_setup()
                
                case "MAIN_ACTIVITY":
                    return self._handle_main_activity()
                
                case "BANK_AGAIN":
                    return self._handle_bank_again()
                
                case _:
                    logging.warning(f"[{self.id}] Unknown phase: {phase}")
                    return self.loop_interval_ms
        
        except Exception as e:
            logging.error(f"[{self.id}] Error in phase {phase}: {e}")
            return self.loop_interval_ms
    
    def _handle_bank_setup(self) -> int:
        """Handle initial bank setup using the bank plan."""
        bank_status = self.bank_plan.loop(ui)
        
        if bank_status == BankPlan.SUCCESS:
            logging.info(f"[{self.id}] Bank setup completed successfully!")
            self.set_phase("MAIN_ACTIVITY")
            return self.loop_interval_ms
        
        elif bank_status == BankPlan.MISSING_ITEMS:
            error_msg = self.bank_plan.get_error_message()
            logging.error(f"[{self.id}] Bank setup failed - missing items: {error_msg}")
            # You could handle this by going to GE or stopping the plan
            return self.loop_interval_ms
        
        elif bank_status == BankPlan.ERROR:
            error_msg = self.bank_plan.get_error_message()
            logging.error(f"[{self.id}] Bank setup error: {error_msg}")
            return self.loop_interval_ms
        
        else:
            # Still working on bank setup (TRAVELING, BANKING, EQUIPPING, etc.)
            return bank_status
    
    def _handle_main_activity(self) -> int:
        """Handle the main activity after bank setup."""
        # Example: Check if inventory is full and need to bank again
        if inventory.is_full():
            logging.info(f"[{self.id}] Inventory full, going back to bank...")
            self.bank_plan.reset()  # Reset bank plan for reuse
            self.set_phase("BANK_AGAIN")
            return self.loop_interval_ms
        
        # Example main activity logic here
        logging.info(f"[{self.id}] Doing main activity...")
        time.sleep(1)  # Simulate some work
        return self.loop_interval_ms
    
    def _handle_bank_again(self) -> int:
        """Handle banking again when inventory is full."""
        bank_status = self.bank_plan.loop(ui)
        
        if bank_status == BankPlan.SUCCESS:
            logging.info(f"[{self.id}] Banking completed, returning to main activity!")
            self.set_phase("MAIN_ACTIVITY")
            return self.loop_interval_ms
        
        elif bank_status == BankPlan.MISSING_ITEMS:
            error_msg = self.bank_plan.get_error_message()
            logging.error(f"[{self.id}] Banking failed - missing items: {error_msg}")
            return self.loop_interval_ms
        
        elif bank_status == BankPlan.ERROR:
            error_msg = self.bank_plan.get_error_message()
            logging.error(f"[{self.id}] Banking error: {error_msg}")
            return self.loop_interval_ms
        
        else:
            # Still working on banking
            return bank_status


# Example of how to create different bank configurations
def create_combat_bank_plan():
    """Create a bank plan configured for combat activities."""
    return BankPlan(
        bank_area="FALADOR_BANK",
        food_item="Trout",
        food_quantity=25,
        inventory_config={
            "required_items": ["Teleport to house", "Coins"],
            "optional_items": ["Prayer potion(4)"],
            "deposit_all": True
        }
    )


def create_skilling_bank_plan():
    """Create a bank plan configured for skilling activities."""
    return BankPlan(
        bank_area="VARROCK_WEST",
        food_item="Bread",
        food_quantity=10,
        inventory_config={
            "required_items": ["Knife", "Tinderbox"],
            "optional_items": [],
            "deposit_all": True
        }
    )
