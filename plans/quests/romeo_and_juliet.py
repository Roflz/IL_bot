#!/usr/bin/env python3
"""
Romeo and Juliet Quest Plan (Version 2)
=======================================

This plan uses the modular BankPlan and GePlan utilities for quest item setup,
following the same pattern as woodcutting.py and falador_cows.py.

The plan handles:
1. Banking setup for quest items
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
from typing import Optional
from pathlib import Path

# Add the parent directory to the path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from actions import objects, player, chat, inventory, bank
from actions import npc, travel
from actions import get_player_plane
from actions import wait_until
from constants import REGIONS
from helpers import quest as quest_helper
from helpers.npc import closest_npc_by_name
from helpers.keyboard import press_esc
from plans.utilities.bank_plan_simple import BankPlanSimple
from plans.utilities.ge import GePlan, create_ge_plan
from plans.base import Plan


class RomeoAndJuliet2Plan(Plan):
    """Romeo and Juliet quest plan using modular utilities."""
    
    id = "ROMEO_AND_JULIET_2"
    label = "Quest: Romeo & Juliet (Modular)"
    description = """Completes the Romeo and Juliet quest using modular bank and GE utilities. Automatically acquires quest items and progresses through all quest phases with proper error handling.

Starting Area: Any bank (travels to quest locations)
Required Items: Quest items (purchased via GE if needed)"""
    
    # Return status codes
    SUCCESS = 0
    BANK_SETUP = 1
    MISSING_ITEMS = 2
    QUEST_PROGRESS = 3
    ERROR = 4
    
    def __init__(self):
        """Initialize the Romeo and Juliet quest plan."""
        self.state = {"phase": "BANK_SETUP"}
        self.next = self.state["phase"]
        self.loop_interval_ms = 600
        
        # Quest item configuration
        self.quest_items = ["Cadava berries"]
        
        # Bank plan for quest setup
        self.bank_plan = BankPlanSimple(
            bank_area="VARROCK_WEST",
            required_items=self.quest_items,
            deposit_all=True,
            equip_items={}  # No equipment needed for this quest
        )
        
        # GE plan for buying missing items
        self.ge_plan = None
        
        # GE strategy for quest items
        self.ge_strategy = {
            "Cadava berries": {"quantity": 1, "bumps": 5, "set_price": 0},
            "default": {"quantity": 1, "bumps": 5, "set_price": 0}
        }
        
        # Quest state tracking
        self.quest_completed = False
        self.error_message = None
        
        # Set up camera immediately during initialization
        try:
            from helpers import setup_camera_optimal
            setup_camera_optimal()
        except Exception as e:
            logging.warning(f"[{self.id}] Could not setup camera: {e}")
        
        logging.info(f"[{self.id}] Romeo and Juliet quest plan initialized")
    
    def set_phase(self, phase: str, camera_setup: bool = True):
        """Set the current phase."""
        from helpers import set_phase_with_camera
        return set_phase_with_camera(self, phase, camera_setup)
    
    def loop(self, ui) -> int:
        """Main loop method following standard plan protocol."""
        phase = self.state.get("phase", "BANK_SETUP")
        logged_in = player.logged_in()
        if not logged_in:
            player.login()
            return self.loop_interval_ms

        # Check if quest is already finished
        if quest_helper.quest_finished("Romeo and Juliet") and not phase == 'DONE':
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

            case "START_QUEST":
                return self._handle_start_quest(ui)

            case "TALK_TO_JULIET_1":
                return self._handle_talk_to_juliet_1(ui)

            case "TALK_TO_ROMEO_1":
                return self._handle_talk_to_romeo_1(ui)

            case "FATHER_LAWRENCE":
                return self._handle_father_lawrence(ui)

            case "GET_POTION":
                return self._handle_get_potion(ui)

            case "GIVE_POTION_TO_JULIET":
                return self._handle_give_potion_to_juliet(ui)

            case "FINISH_QUEST":
                return self._handle_finish_quest(ui)

            case "QUEST_COMPLETE":
                return self._handle_quest_complete(ui)

            case "DONE":
                return self._handle_done()

        logging.warning(f"[{self.id}] Unknown phase: {phase}")
        return self.loop_interval_ms


    
    def _handle_bank_setup(self, ui) -> int:
        """Handle bank setup for quest items."""
        logging.info(f"[{self.id}] Setting up bank for quest items...")
        
        # Run the bank plan
        bank_status = self.bank_plan.loop(ui)
        
        if bank_status == BankPlanSimple.SUCCESS:
            logging.info(f"[{self.id}] Bank setup completed successfully")
            if bank.is_open():
                bank.close_bank()
            self.set_phase("START_QUEST")
            return self.QUEST_PROGRESS
        
        elif bank_status == BankPlanSimple.MISSING_ITEMS:
            logging.info(f"[{self.id}] Missing quest items detected, will try to buy from GE")
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
            self.set_phase("MISSING_ITEMS")
            return self.MISSING_ITEMS
        
        elif bank_status == BankPlanSimple.ERROR:
            error_msg = self.bank_plan.get_error_message()
            logging.error(f"[{self.id}] Bank setup error: {error_msg}")
            self.set_phase("ERROR_STATE")
            self.error_message = f"Bank setup failed: {error_msg}"
            return self.ERROR
        
        else:
            # Still working on bank setup (TRAVELING, BANKING, etc.)
            return bank_status
    
    def _handle_missing_items(self, ui) -> int:
        """Handle buying missing quest items from GE."""
        logging.info(f"[{self.id}] Handling missing quest items...")
        
        # GE plan should have been created in _handle_bank_setup
        if self.ge_plan is None:
            logging.error(f"[{self.id}] No GE plan created, returning to bank")
            self.bank_plan.reset()
            self.set_phase("BANK_SETUP")
            return self.BANK_SETUP
        
        logging.info(f"[{self.id}] Running GE plan to buy quest items...")
        
        # Use the GE plan to buy missing items
        ge_status = self.ge_plan.loop(ui)
        
        if ge_status == GePlan.SUCCESS:
            logging.info(f"[{self.id}] Successfully purchased all missing items!")
            # Reset bank plan and try banking again
            self.bank_plan.reset()
            self.ge_plan = None
            self.set_phase("BANK_SETUP")
            return self.BANK_SETUP
        
        elif ge_status == GePlan.ERROR:
            error_msg = self.ge_plan.get_error_message()
            logging.error(f"[{self.id}] GE plan failed: {error_msg}")
            self.set_phase("ERROR_STATE")
            self.error_message = f"GE purchase failed: {error_msg}"
            return self.ERROR
        
        elif ge_status == GePlan.INSUFFICIENT_FUNDS:
            error_msg = self.ge_plan.get_error_message()
            logging.error(f"[{self.id}] Insufficient funds for GE purchases: {error_msg}")
            self.set_phase("ERROR_STATE")
            self.error_message = f"Insufficient funds for quest items: {error_msg}"
            return self.ERROR
        
        else:
            # Still working on buying items (TRAVELING, CHECKING_COINS, BUYING, WAITING)
            logging.info(f"[{self.id}] GE plan in progress... Status: {ge_status}")
            return ge_status
    
    def _handle_start_quest(self, ui) -> int:
        """Handle starting the quest."""
        logging.info(f"[{self.id}] Starting Romeo and Juliet quest...")
        
        # Check if quest is already in progress
        if quest_helper.quest_in_progress("Romeo & Juliet"):
            logging.info(f"[{self.id}] Quest already in progress")
            self.set_phase("TALK_TO_JULIET_1")
            return self.QUEST_PROGRESS
        
        # Check if we have the required items
        from actions import inventory
        if not inventory.has_item("Cadava berries"):
            logging.warning(f"[{self.id}] Missing Cadava berries, going back to bank setup")
            self.bank_plan.reset()
            self.set_phase("BANK_SETUP")
            return self.BANK_SETUP
        
        # Go to Romeo to start the quest
        if not travel.in_area(REGIONS["VARROCK_SQUARE"]) and not closest_npc_by_name("Romeo"):
            logging.info(f"[{self.id}] Traveling to Romeo")
            travel.go_to_and_find_npc("VARROCK_SQUARE", "Romeo")
            return self.QUEST_PROGRESS
        
        # Talk to Romeo
        if closest_npc_by_name("Romeo"):
            if not chat.dialogue_is_open() and not chat.can_choose_option():
                logging.info(f"[{self.id}] Talking to Romeo")
                npc.click_npc_action("Romeo", "Talk-to")
                wait_until(chat.dialogue_is_open, max_wait_ms=4000)
                return self.QUEST_PROGRESS
            else:
                if chat.dialogue_is_open():
                    chat.continue_dialogue()
                    return self.QUEST_PROGRESS
                if chat.can_choose_option():
                    if chat.option_exists("Perhaps I could help"):
                        chat.choose_option("Perhaps I could help")
                        return self.QUEST_PROGRESS
                    elif chat.option_exists("Yes."):
                        chat.choose_option("Yes.")
                        return self.QUEST_PROGRESS
        
        return self.QUEST_PROGRESS
    
    def _handle_talk_to_juliet_1(self, ui) -> int:
        """Handle talking to Juliet for the first time."""
        logging.info(f"[{self.id}] Talking to Juliet...")
        
        from actions import inventory
        if inventory.has_item("Message"):
            logging.info(f"[{self.id}] Got message from Juliet")
            time.sleep(0.6)
            self.set_phase("TALK_TO_ROMEO_1")
            return self.QUEST_PROGRESS
        
        # Go to Juliet's mansion
        if not travel.in_area(REGIONS["JULIET_MANSION"]) and get_player_plane() == 0:
            logging.info(f"[{self.id}] Traveling to Juliet's mansion")
            travel.go_to("JULIET_MANSION")
            return self.QUEST_PROGRESS
        
        # Climb up to Juliet's room
        if get_player_plane() == 0:
            logging.info(f"[{self.id}] Climbing up to Juliet's room")
            objects.click_object_closest_by_distance_simple("Staircase", "Climb-up")
            wait_until(lambda: get_player_plane() == 1, max_wait_ms=5000)
            return self.QUEST_PROGRESS
        
        # Talk to Juliet
        if get_player_plane() == 1 and not chat.dialogue_is_open():
            logging.info(f"[{self.id}] Talking to Juliet")
            npc.click_npc_action("Juliet", "Talk-to")
            return self.QUEST_PROGRESS
        elif chat.dialogue_is_open():
            chat.continue_dialogue()
            return self.QUEST_PROGRESS
        
        return self.QUEST_PROGRESS
    
    def _handle_talk_to_romeo_1(self, ui) -> int:
        """Handle talking to Romeo after getting message from Juliet."""
        logging.info(f"[{self.id}] Talking to Romeo with message...")
        
        if not inventory.has_item("Message"):
            chat.continue_dialogue()
            self.set_phase("FATHER_LAWRENCE")
            return self.QUEST_PROGRESS
        
        # Go down from Juliet's room
        if get_player_plane() == 1:
            logging.info(f"[{self.id}] Going down from Juliet's room")
            objects.click_object_closest_by_distance_simple("Staircase", "Climb-down")
            wait_until(lambda: get_player_plane() == 0, max_wait_ms=5000)
            return self.QUEST_PROGRESS
        
        # Go to Romeo
        if not travel.in_area(REGIONS["VARROCK_SQUARE"]):
            logging.info(f"[{self.id}] Traveling to Romeo")
            travel.go_to_and_find_npc("VARROCK_SQUARE", "Romeo")
            return self.QUEST_PROGRESS
        
        # Talk to Romeo
        if not chat.dialogue_is_open() and not chat.can_continue():
            logging.info(f"[{self.id}] Talking to Romeo")
            npc.click_npc_action("Romeo", "Talk-to")
            return self.QUEST_PROGRESS
        else:
            chat.continue_dialogue()
            return self.QUEST_PROGRESS
    
    def _handle_father_lawrence(self, ui) -> int:
        """Handle talking to Father Lawrence."""
        logging.info(f"[{self.id}] Talking to Father Lawrence...")
        
        if chat.dialogue_contains("Apart from the strong overtones") or chat.dialogue_contains("Ah, have you found the Apothecary yet?"):
            chat.continue_dialogue()
            self.set_phase("GET_POTION")
            return self.QUEST_PROGRESS
        
        # Go to Father Lawrence
        if not travel.in_area(REGIONS["VARROCK_CHURCH"]):
            logging.info(f"[{self.id}] Traveling to Father Lawrence")
            travel.go_to("VARROCK_CHURCH")
            return self.QUEST_PROGRESS
        
        # Talk to Father Lawrence
        if not chat.dialogue_is_open() and not chat.can_continue() and not player.in_cutscene():
            logging.info(f"[{self.id}] Talking to Father Lawrence")
            npc.click_npc_action("Father Lawrence", "Talk-to")
            return self.QUEST_PROGRESS
        else:
            chat.continue_dialogue()
            return self.QUEST_PROGRESS
    
    def _handle_get_potion(self, ui) -> int:
        """Handle getting the potion from the Apothecary."""
        logging.info(f"[{self.id}] Getting potion from Apothecary...")
        
        from actions import inventory
        if inventory.has_item("Cadava potion"):
            logging.info(f"[{self.id}] Got Cadava potion")
            self.set_phase("GIVE_POTION_TO_JULIET")
            return self.QUEST_PROGRESS
        
        # Talk to Apothecary
        if closest_npc_by_name("Apothecary"):
            if not chat.dialogue_is_open() and not chat.can_continue() and not chat.get_options():
                logging.info(f"[{self.id}] Talking to Apothecary")
                npc.click_npc_action("Apothecary", "Talk-to")
                return self.QUEST_PROGRESS
            elif chat.can_choose_option():
                if chat.option_exists("Talk about something else."):
                    chat.choose_option("Talk about something else.")
                    return self.QUEST_PROGRESS
                elif chat.option_exists("Talk about Romeo & Juliet."):
                    chat.choose_option("Talk about Romeo & Juliet.")
                    return self.QUEST_PROGRESS
            else:
                chat.continue_dialogue()
                return self.QUEST_PROGRESS
        
        # Go to Apothecary
        elif not travel.in_area(REGIONS["VARROCK_APOTHECARY"]):
            logging.info(f"[{self.id}] Traveling to Apothecary")
            travel.go_to("VARROCK_APOTHECARY")
            return self.QUEST_PROGRESS
        
        return self.QUEST_PROGRESS
    
    def _handle_give_potion_to_juliet(self, ui) -> int:
        """Handle giving the potion to Juliet."""
        logging.info(f"[{self.id}] Giving potion to Juliet...")
        
        from actions import inventory
        if not inventory.has_item("Cadava potion"):
            logging.info(f"[{self.id}] Potion given to Juliet")
            self.set_phase("FINISH_QUEST")
            return self.QUEST_PROGRESS
        
        # Go to Juliet's mansion
        if not travel.in_area(REGIONS["JULIET_MANSION"]) and get_player_plane() == 0:
            logging.info(f"[{self.id}] Traveling to Juliet's mansion")
            travel.go_to("JULIET_MANSION")
            return self.QUEST_PROGRESS
        
        # Climb up to Juliet's room
        if get_player_plane() == 0:
            logging.info(f"[{self.id}] Climbing up to Juliet's room")
            objects.click_object_closest_by_distance_simple("Staircase", "Climb-up")
            wait_until(lambda: get_player_plane() == 1, max_wait_ms=5000)
            return self.QUEST_PROGRESS
        
        # Talk to Juliet
        if get_player_plane() == 1 and not chat.can_continue() and not player.in_cutscene():
            logging.info(f"[{self.id}] Talking to Juliet")
            npc.click_npc_action("Juliet", "Talk-to")
            return self.QUEST_PROGRESS
        elif chat.can_continue() and not chat.dialogue_contains("Please go to Romeo and make sure he understands."):
            chat.continue_dialogue()
            return self.QUEST_PROGRESS
        elif chat.can_continue():
            chat.continue_dialogue()
            wait_until(player.in_cutscene, max_wait_ms=3000)
            return self.QUEST_PROGRESS
        
        return self.QUEST_PROGRESS
    
    def _handle_finish_quest(self, ui) -> int:
        """Handle finishing the quest."""
        logging.info(f"[{self.id}] Finishing quest...")
        
        if quest_helper.quest_state("Romeo & Juliet") == "FINISHED":
            logging.info(f"[{self.id}] Quest completed!")
            self.set_phase("QUEST_COMPLETE")
            return self.SUCCESS
        
        # Go down from Juliet's room
        if get_player_plane() == 1 and not player.in_cutscene():
            logging.info(f"[{self.id}] Going down from Juliet's room")
            objects.click_object_closest_by_distance_simple("Staircase", "Climb-down")
            wait_until(lambda: get_player_plane() == 0, max_wait_ms=5000)
            return self.QUEST_PROGRESS
        
        # Go to Romeo
        if not travel.in_area(REGIONS["VARROCK_SQUARE"]) and not player.in_cutscene():
            logging.info(f"[{self.id}] Traveling to Romeo")
            travel.go_to_and_find_npc("VARROCK_SQUARE", "Romeo")
            return self.QUEST_PROGRESS
        
        # Handle cutscene
        if player.in_cutscene():
            chat.continue_dialogue()
            return self.QUEST_PROGRESS
        elif chat.dialogue_is_open():
            chat.continue_dialogue()
            return 2000
        
        # Talk to Romeo
        if wait_until(lambda: not chat.dialogue_is_open() and not chat.can_continue() and not player.in_cutscene(), max_wait_ms=600):
            logging.info(f"[{self.id}] Talking to Romeo")
            npc.click_npc_action("Romeo", "Talk-to")
            return self.QUEST_PROGRESS

        
        return self.QUEST_PROGRESS
    
    def _handle_quest_complete(self, ui) -> int:
        """Handle quest completion."""
        logging.info(f"[{self.id}] Quest completed successfully!")
        self.quest_completed = True
        press_esc()
        self.set_phase("DONE")
        return self.SUCCESS
    
    def _handle_done(self) -> int:
        """Handle done phase - plan is complete."""
        logging.info(f"[{self.id}] Plan completed - staying in DONE phase")
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
