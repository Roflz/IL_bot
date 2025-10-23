#!/usr/bin/env python3
"""
Goblin Diplomacy Quest Plan (Version 2)
=======================================

This plan uses the modular BankPlan and GePlan utilities for quest item setup,
following the same pattern as romeo_and_juliet_2.py and falador_cows_2.py.

The plan handles:
1. Banking setup for quest items (Blue dye, Orange dye, Goblin mail)
2. GE buying for missing quest items
3. Quest progression through all phases
4. Proper error handling and phase management

Return Status Codes:
- 0: SUCCESS - Quest completed successfully
- 1: BANK_SETUP - Setting up bank for quest items
- 2: MISSING_ITEMS - Need to buy quest items from GE
- 3: QUEST_PROGRESS - Working through quest phases
- 4: ERROR - An error occurred
"""

import time
import logging
from typing import Dict, List, Optional
from pathlib import Path

# Add the parent directory to the path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from ..actions import objects, player, npc, chat, travel, inventory, tab, bank
from ..actions.player import get_player_plane
from ..actions.timing import wait_until
from ..actions.bank import close_bank
from ..constants import REGIONS
from ..helpers import quest as quest_helper
from ..helpers.npc import closest_npc_by_name
from ..helpers.utils import press_esc
from .utilities.bank_plan import BankPlan
from .utilities.ge import GePlan, create_ge_plan
from .base import Plan


class GoblinDiplomacy2Plan(Plan):
    """Goblin Diplomacy quest plan using modular utilities."""
    
    id = "GOBLIN_DIPLOMACY_2"
    label = "Quest: Goblin Diplomacy (Modular)"
    
    # Return status codes
    SUCCESS = 0
    BANK_SETUP = 1
    MISSING_ITEMS = 2
    QUEST_PROGRESS = 3
    ERROR = 4
    
    def __init__(self):
        """Initialize the Goblin Diplomacy quest plan."""
        self.state = {"phase": "BANK_SETUP"}
        self.next = self.state["phase"]
        self.loop_interval_ms = 600
        
        # Quest item configuration
        self.quest_items = ["Blue dye", "Orange dye", "Goblin mail"]
        
        # Bank plan for quest setup
        self.bank_plan = BankPlan(
            bank_area="CLOSEST_BANK",  # Use closest bank for convenience
            food_item=None,  # No food needed for this quest
            food_quantity=0,
            equipment_config={
                "weapon_tiers": [],  # No weapons needed for this quest
                "armor_tiers": {},   # No armor needed for this quest
                "jewelry_tiers": {}, # No jewelry needed for this quest
                "tool_tiers": []     # No tools needed for this quest
            },
            inventory_config={
                "required_items": [
                    "Blue dye",
                    "Orange dye", 
                    {"name": "Goblin mail", "quantity": 3}
                ],
                "optional_items": [],
                "deposit_all": True
            }
        )
        
        # GE plan for buying missing items
        self.ge_plan = None
        
        # GE strategy for quest items
        self.ge_strategy = {
            "Blue dye": {"quantity": 1, "bumps": 5, "set_price": 3000},
            "Orange dye": {"quantity": 1, "bumps": 5, "set_price": 3000},
            "Goblin mail": {"quantity": 3, "bumps": 20, "set_price": 1000},
            "default": {"quantity": 1, "bumps": 5, "set_price": 1000}
        }
        
        # Quest state tracking
        self.quest_completed = False
        self.error_message = None
        
        # Set up camera immediately during initialization
        try:
            from ..helpers.camera import setup_camera_optimal
            setup_camera_optimal()
        except Exception as e:
            logging.warning(f"[{self.id}] Could not setup camera: {e}")
        
        logging.info(f"[{self.id}] Goblin Diplomacy quest plan initialized")
    
    def set_phase(self, phase: str, camera_setup: bool = True):
        """Set the current phase."""
        from ..helpers.phase_utils import set_phase_with_camera
        return set_phase_with_camera(self, phase, camera_setup)
    
    def loop(self, ui) -> int:
        """Main loop method following standard plan protocol."""
        phase = self.state.get("phase", "BANK_SETUP")
        logged_in = player.logged_in()
        if not logged_in:
            player.login()
            return self.loop_interval_ms

        # Check if quest is already finished
        if quest_helper.quest_finished("Goblin Diplomacy") and not phase == 'DONE':
            if chat.can_continue():
                chat.continue_dialogue()
                return self.QUEST_PROGRESS
            press_esc()
            self.set_phase('DONE')
            return self.SUCCESS

        match(phase):
            case "BANK_SETUP":
                return self._handle_bank_setup(ui)

            case "MISSING_ITEMS":
                return self._handle_missing_items(ui)

            case "MAKE_ARMOURS":
                return self._handle_make_armours(ui)

            case "START_QUEST":
                return self._handle_start_quest(ui)

            case "TALK_TO_GENERAL_WARTFACE":
                return self._handle_talk_to_general_wartface(ui)

            case "DONE":
                return self._handle_done(ui)

        logging.warning(f"[{self.id}] Unknown phase: {phase}")
        return self.loop_interval_ms
    
    def _handle_bank_setup(self, ui) -> int:
        """Handle bank setup for quest items."""
        logging.info(f"[{self.id}] Setting up bank for quest items...")
        
        # Run the bank plan
        bank_status = self.bank_plan.loop(ui)
        
        if bank_status == BankPlan.SUCCESS:
            logging.info(f"[{self.id}] Bank setup completed successfully")
            self.set_phase("MAKE_ARMOURS")
            return self.QUEST_PROGRESS
        
        elif bank_status == BankPlan.MISSING_ITEMS:
            logging.info(f"[{self.id}] Missing quest items, transitioning to GE")
            self.set_phase("MISSING_ITEMS")
            return self.MISSING_ITEMS
        
        elif bank_status == BankPlan.ITEMS_TO_SELL:
            # Quest plans don't need to sell items, treat as success
            logging.info(f"[{self.id}] Bank found items to sell, but quest doesn't need selling - treating as success")
            self.set_phase("MAKE_ARMOURS")
            return self.QUEST_PROGRESS
        
        elif bank_status == BankPlan.ERROR:
            logging.error(f"[{self.id}] Bank setup failed: {self.bank_plan.get_error_message()}")
            self.set_phase("ERROR_STATE")
            self.error_message = f"Bank setup failed: {self.bank_plan.get_error_message()}"
            return self.ERROR
        
        else:
            # Still working on bank setup
            return self.BANK_SETUP
    
    def _handle_missing_items(self, ui) -> int:
        """Handle buying missing quest items from GE."""
        logging.info(f"[{self.id}] Handling missing quest items...")
        
        # Parse missing items from bank plan
        missing_items = []
        error_msg = self.bank_plan.get_error_message()
        if error_msg and "Missing required items:" in error_msg:
            # Extract item names from error message
            items_str = error_msg.split("Missing required items: ")[1]
            items_str = items_str.strip("[]")
            missing_items = [item.strip().strip("'\"") for item in items_str.split(",")]
        
        if not missing_items:
            logging.warning(f"[{self.id}] No missing items found in error message")
            self.set_phase("ERROR_STATE")
            self.error_message = "Could not determine missing items"
            return self.ERROR
        
        # Create GE plan if not already created
        if self.ge_plan is None:
            logging.info(f"[{self.id}] Creating GE plan for items: {missing_items}")
            
            # Create items list with GE strategy (correct format)
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
            
            self.ge_plan = create_ge_plan(items_to_buy)
        
        # Run the GE plan
        ge_status = self.ge_plan.loop(ui)
        
        if ge_status == GePlan.SUCCESS:
            logging.info(f"[{self.id}] GE purchase completed successfully")
            # Reset bank plan and go back to bank setup
            self.bank_plan.reset()
            self.ge_plan = None
            self.set_phase("BANK_SETUP")
            return self.BANK_SETUP
        
        elif ge_status == GePlan.ERROR:
            logging.error(f"[{self.id}] GE purchase failed")
            self.set_phase("ERROR_STATE")
            self.error_message = "Failed to purchase quest items from GE"
            return self.ERROR
        
        elif ge_status == GePlan.INSUFFICIENT_FUNDS:
            error_msg = self.ge_plan.get_error_message()
            logging.error(f"[{self.id}] Insufficient funds for GE purchases: {error_msg}")
            self.set_phase("ERROR_STATE")
            self.error_message = f"Insufficient funds for quest items: {error_msg}"
            return self.ERROR
        
        else:
            # Still working on GE purchase
            return self.MISSING_ITEMS
    
    def _handle_make_armours(self, ui) -> int:
        """Handle making the dyed goblin mail armours."""
        logging.info(f"[{self.id}] Making dyed goblin mail armours...")
        
        # Ensure bank is closed
        if bank.is_open():
            bank.close_bank()
            return self.QUEST_PROGRESS
        
        # Ensure inventory tab is open
        if not tab.is_tab_open("INVENTORY"):
            tab.open_tab("INVENTORY")
            return self.QUEST_PROGRESS
        
        
        # Use blue dye on one goblin mail
        if not inventory.has_item("Blue goblin mail"):
            logging.info(f"[{self.id}] Using blue dye on goblin mail")
            result = inventory.use_item_on_item("Blue dye", "Goblin mail")
            if result is not None:
                return 2000  # Wait for the action to complete
        
        # Use orange dye on another goblin mail
        if not inventory.has_item("Orange goblin mail"):
            logging.info(f"[{self.id}] Using orange dye on goblin mail")
            result = inventory.use_item_on_item("Orange dye", "Goblin mail")
            if result is not None:
                return 2000  # Wait for the action to complete
        
        # Check if both armours are made
        if inventory.has_item("Blue goblin mail") and inventory.has_item("Orange goblin mail"):
            logging.info(f"[{self.id}] Both armours made successfully")
            self.set_phase("START_QUEST")
            return self.QUEST_PROGRESS
        
        return self.QUEST_PROGRESS
    
    def _handle_start_quest(self, ui) -> int:
        """Handle starting the quest."""
        logging.info(f"[{self.id}] Starting Goblin Diplomacy quest...")
        
        # Check if quest is already in progress
        if quest_helper.quest_in_progress("Goblin Diplomacy"):
            logging.info(f"[{self.id}] Quest already in progress")
            self.set_phase("TALK_TO_GENERAL_WARTFACE")
            return self.QUEST_PROGRESS
        
        # Go to Goblin Village to start the quest
        if not travel.in_area("GOBLIN_VILLAGE") or not closest_npc_by_name("General Bentnoze"):
            logging.info(f"[{self.id}] Traveling to Goblin Village")
            travel.go_to("GOBLIN_VILLAGE", center=True)
            return self.QUEST_PROGRESS
        
        # Talk to General Bentnoze
        if closest_npc_by_name("General Bentnoze"):
            if not chat.dialogue_is_open() and not chat.can_choose_option():
                logging.info(f"[{self.id}] Talking to General Bentnoze")
                npc.click_npc_action("General Bentnoze", "Talk-to")
                wait_until(chat.dialogue_is_open, max_wait_ms=4000)
                return self.QUEST_PROGRESS
            elif chat.can_choose_option():
                if chat.option_exists("Do you want me to pick an armour"):
                    chat.choose_option("Do you want me to pick an armour")
                    return self.QUEST_PROGRESS
                elif chat.option_exists("What about a different colour"):
                    chat.choose_option("What about a different colour")
                    return self.QUEST_PROGRESS
                elif chat.option_exists("Yes, he looks fat."):
                    chat.choose_option("Yes, he looks fat.")
                    return self.QUEST_PROGRESS
                elif chat.option_exists("Yes."):
                    chat.choose_option("Yes.")
                    return self.QUEST_PROGRESS
            else:
                chat.continue_dialogue()
                return self.QUEST_PROGRESS
        
        return self.QUEST_PROGRESS
    
    def _handle_talk_to_general_wartface(self, ui) -> int:
        """Handle talking to General Wartface."""
        logging.info(f"[{self.id}] Talking to General Wartface...")
        
        # Check if quest is finished
        if quest_helper.quest_finished("Goblin Diplomacy"):
            logging.info(f"[{self.id}] Quest finished!")
            self.set_phase("DONE")
            return self.SUCCESS
        
        # Go to Goblin Village if not there
        if not travel.in_area("GOBLIN_VILLAGE") and not closest_npc_by_name("General Bentnoze"):
            logging.info(f"[{self.id}] Traveling to Goblin Village")
            travel.go_to("GOBLIN_VILLAGE", center=True)
            return self.QUEST_PROGRESS
        
        # Talk to General Wartface
        if closest_npc_by_name("General Wartface") and (not player.in_cutscene() or chat.can_continue()):
            if not chat.dialogue_is_open() and not chat.can_choose_option():
                logging.info(f"[{self.id}] Talking to General Wartface")
                npc.click_npc_action("General Wartface", "Talk-to")
                wait_until(chat.dialogue_is_open, max_wait_ms=4000)
                return self.QUEST_PROGRESS
            elif chat.can_choose_option():
                if chat.option_exists("I have some orange armour here."):
                    chat.choose_option("I have some orange armour here.")
                    return self.QUEST_PROGRESS
                elif chat.option_exists("I have some blue armour here."):
                    chat.choose_option("I have some blue armour here.")
                    return self.QUEST_PROGRESS
                elif chat.option_exists("I have some brown armour here."):
                    chat.choose_option("I have some brown armour here.")
                    return self.QUEST_PROGRESS
                elif chat.option_exists("Yes, he looks fat."):
                    chat.choose_option("Yes, he looks fat.")
                    return self.QUEST_PROGRESS
            else:
                chat.continue_dialogue()
                return self.QUEST_PROGRESS
        
        return self.QUEST_PROGRESS
    
    def _handle_done(self, ui) -> int:
        """Handle quest completion."""
        logging.info(f"[{self.id}] Quest completed successfully!")
        self.quest_completed = True
        press_esc()
        return self.SUCCESS
    
    def _handle_error_state(self, ui) -> int:
        """Handle error state."""
        logging.error(f"[{self.id}] Error: {self.error_message}")
        return self.ERROR
    
    def is_quest_complete(self) -> bool:
        """Check if quest is completed."""
        return self.quest_completed
    
    def get_error_message(self) -> Optional[str]:
        """Get the current error message."""
        return self.error_message
    
    def reset(self):
        """Reset the plan to initial state."""
        self.state = {"phase": "BANK_SETUP"}
        self.quest_completed = False
        self.error_message = None
        self.bank_plan.reset()
        self.ge_plan = None
        logging.info(f"[{self.id}] Plan reset to initial state")
