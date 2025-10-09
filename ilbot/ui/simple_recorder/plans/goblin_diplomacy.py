# goblin_diplomacy.py
import time
import logging

from ..actions import objects, player, tab
from ..actions.player import get_player_plane
from ..actions.timing import wait_until
from ..constants import BANK_REGIONS, REGIONS
import ilbot.ui.simple_recorder.actions.travel as trav
import ilbot.ui.simple_recorder.actions.bank as bank
import ilbot.ui.simple_recorder.actions.inventory as inv
import ilbot.ui.simple_recorder.actions.ge as ge
import ilbot.ui.simple_recorder.actions.npc as npc
import ilbot.ui.simple_recorder.actions.chat as chat

from .base import Plan
from ..helpers import quest
from ..helpers.bank import near_any_bank
from ..helpers.utils import press_esc


class GoblinDiplomacyPlan(Plan):
    id = "GOBLIN_DIPLOMACY"
    label = "Quest: Goblin Diplomacy"

    def __init__(self):
        self.state = {"phase": "START_QUEST"}  # gate: ensure items first
        self.next = self.state["phase"]
        self.loop_interval_ms = 600
        
        # Set up camera immediately during initialization
        from ilbot.ui.simple_recorder.helpers.camera import setup_camera_optimal
        setup_camera_optimal()


    def set_phase(self, phase: str, camera_setup: bool = True):
        from ..helpers.phase_utils import set_phase_with_camera
        return set_phase_with_camera(self, phase, camera_setup)

    def loop(self, ui):
        phase = self.state.get("phase", "GO_TO_CLOSEST_BANK")
        if quest.quest_finished("Goblin Diplomacy") and not phase == 'DONE':
            if chat.can_continue():
                chat.continue_dialogue()
                return
            press_esc()
            self.set_phase('DONE', ui)
            return

        match(phase):
            case "GO_TO_CLOSEST_BANK":
                if not near_any_bank():
                    trav.go_to_closest_bank()
                else:
                    self.set_phase("CHECK_BANK_FOR_QUEST_ITEMS", ui)
                return

            case "CHECK_BANK_FOR_QUEST_ITEMS":
                if bank.is_closed():
                    bank.open_bank()
                    wait_until(lambda: bank.is_open(), max_wait_ms=5000, min_wait_ms=200)
                    bank.deposit_inventory()
                    wait_until(inv.is_empty, max_wait_ms=5000, min_wait_ms=200)
                    return
                elif bank.is_open():
                    
                    # Now check what we have in the bank and what we need
                    required_items = ["Blue dye", "Orange dye", "Goblin mail"]
                    needed_items = []
                    
                    missing_items = []
                    for item in required_items:
                        # Check if bank has the item
                        if bank.has_item(item):
                            needed_items.append(item)
                        else:
                            # Item not in bank, need to buy from GE
                            missing_items.append(item)
                            logging.info(f"[GD] Item {item} not found in bank, need to buy from GE")
                    
                    if missing_items:
                        # Some items are missing, need to buy from GE
                        bank.close_bank()
                        if not wait_until(bank.is_closed, min_wait_ms=600, max_wait_ms=3000):
                            return
                        self.set_phase("BUY_QUEST_ITEMS_FROM_GE", ui)
                        return
                    elif needed_items:
                        # Ensure bank note mode is disabled (withdraw as items, not notes)
                        bank.ensure_note_mode_disabled()
                        
                        # Withdraw needed items (as unnoted items)
                        for item in needed_items:
                            if item == "Goblin mail":
                                # Need 3 goblin mail
                                bank.withdraw_item(item, withdraw_x=3)
                            else:
                                bank.withdraw_item(item)

                        # Verify we have all items in inventory before proceeding
                        time.sleep(0.5)  # Brief wait for items to appear in inventory
                        
                        # Check if we have all required items in inventory (unnoted)
                        has_all_required = True
                        for item in required_items:
                            if item == "Goblin mail":
                                # Need 3 goblin mail
                                if not inv.inv_has(item, min_qty=3):
                                    has_all_required = False
                                    break
                            else:
                                if not inv.has_unnoted_item(item):
                                    has_all_required = False
                                    break
                        
                        if has_all_required:
                            bank.close_bank()
                            if wait_until(lambda: bank.is_closed(), min_wait_ms=600, max_wait_ms=3000):
                                self.set_phase("MAKE_ARMOURS", ui)
                                return
                            return
                        else:
                            # Items not properly withdrawn, try again next tick
                            return
                    else:
                        # All items are in bank, but we need to withdraw them
                        # This shouldn't happen since we just checked, but handle it
                        logging.info("[GD] All items found in bank, withdrawing...")
                        bank.ensure_note_mode_disabled()
                        
                        for item in required_items:
                            if item == "Goblin mail":
                                bank.withdraw_item(item, withdraw_x=3)
                            else:
                                bank.withdraw_item(item)
                        
                        # Verify withdrawal and close bank
                        time.sleep(0.5)
                        bank.close_bank()
                        if not wait_until(bank.is_closed, min_wait_ms=600, max_wait_ms=3000):
                            return
                        self.set_phase("MAKE_ARMOURS", ui)
                        return

            case "BUY_QUEST_ITEMS_FROM_GE":
                if not trav.in_area("GE"):
                    trav.go_to("GE")
                    return
                # Define required items
                required_items = ["Blue dye", "Orange dye", "Goblin mail"]
                
                # Check if we have all required items with correct quantities and types
                has_all_items = True
                for item in required_items:
                    if item == "Goblin mail":
                        # Need 3 goblin mail
                        if not inv.has_item(item, min_qty=3):
                            has_all_items = False
                            break
                    else:
                        # Need 1 of each dye (unnoted)
                        if not inv.has_unnoted_item(item):
                            has_all_items = False
                            break
                
                if has_all_items:
                    if ge.is_open():
                        ge.close_ge()
                    else:
                        self.set_phase("CHECK_BANK_FOR_QUEST_ITEMS", ui)
                    return
                
                # Open bank and deposit all inventory first
                if bank.is_closed() and not inv.inv_has("coins"):
                    bank.open_bank()
                    return
                if inv.inv_has("coins") and bank.is_open():
                    bank.close_bank()
                    return
                
                # Deposit all inventory items to get accurate counts
                if bank.is_open():
                    bank.deposit_inventory()
                    time.sleep(0.5)  # Brief wait for deposit to complete
                
                # Check if we already calculated items_to_buy in a previous loop
                if "items_to_buy" not in self.state:
                    # Now count what we have in the bank after depositing
                    items_to_buy = {}
                    
                    # Check Blue dye (need 1 unnoted)
                    blue_dye_count = bank.get_item_count("Blue dye")
                    if blue_dye_count == 0:
                        items_to_buy["Blue dye"] = (1, 5)
                    
                    # Check Orange dye (need 1 unnoted)
                    orange_dye_count = bank.get_item_count("Orange dye")
                    if orange_dye_count == 0:
                        items_to_buy["Orange dye"] = (1, 5)
                    
                    # Check Goblin mail (need 3 total)
                    current_goblin_mail = bank.get_item_count("Goblin mail")
                    needed_goblin_mail = max(0, 3 - current_goblin_mail)
                    if needed_goblin_mail > 0:
                        items_to_buy["Goblin mail"] = (needed_goblin_mail, 20)
                    
                    # Save items_to_buy to state for future loops
                    self.state["items_to_buy"] = items_to_buy
                    logging.info(f"[GD] Calculated items to buy: {items_to_buy}")
                else:
                    # Use previously calculated items_to_buy
                    items_to_buy = self.state["items_to_buy"]
                    logging.info(f"[GD] Using cached items to buy: {items_to_buy}")
                
                # If we don't need to buy anything, move to next phase
                if not items_to_buy:
                    logging.info("[GD] Already have all required items")
                    if ge.is_open():
                        ge.close_ge()
                    else:
                        self.set_phase("CHECK_BANK_FOR_QUEST_ITEMS", ui)
                    return
                
                # Use the centralized GE buying method
                logging.info("[GD] Buying quest items from GE...")
                result = ge.buy_item_from_ge(items_to_buy, ui)
                if result is None:
                    # Still working on buying items, return to continue next tick
                    return
                elif result is True:
                    logging.info("[GD] Successfully bought all quest items")
                else:
                    logging.info("[GD] Failed to buy items, retrying...")
                    return
                
                # If we get here, all items should be bought - verify and go to bank
                time.sleep(0.5)  # Brief wait for items to appear in inventory
                self.set_phase("CHECK_BANK_FOR_QUEST_ITEMS", ui)
                return

            case "MAKE_ARMOURS":
                if not tab.is_tab_open("INVENTORY"):
                    tab.open_tab("INVENTORY")
                    return
                # Use blue dye on one goblin mail
                if not inv.has_item("Blue goblin mail"):
                    result = inv.use_item_on_item("Blue dye", "Goblin mail")
                    if result is not None:
                        return 2000

                # Use orange dye on another goblin mail
                if not inv.has_item("Orange goblin mail"):
                    result = inv.use_item_on_item("Orange dye", "Goblin mail")
                    if result is not None:
                        return 2000

                # Move to next phase when both are dyed
                if inv.has_item("Blue goblin mail") and inv.has_item("Orange goblin mail"):
                    self.set_phase("START_QUEST", ui)
                    return

            case 'START_QUEST':
                if quest.quest_in_progress("Goblin Diplomacy"):
                    self.set_phase("TALK_TO_GENERAL_WARTFACE", ui)
                    return
                    
                # Go to Goblin Village to start the quest
                if not trav.in_area("GOBLIN_VILLAGE") or not npc.closest_npc_by_name("General Bentnoze"):
                    trav.go_to("GOBLIN_VILLAGE", center=True)
                    return
                elif npc.closest_npc_by_name("General Bentnoze"):
                    result = npc.chat_with_npc(
                        "General Bentnoze",
                        options=[
                            "Do you want me to pick an armour",
                            "What about a different colour",
                            "Yes, he looks fat.",
                            "Yes."
                        ]
                    )
                    if result is not None:
                        return result

            case "TALK_TO_GENERAL_WARTFACE":
                if quest.quest_finished("Goblin Diplomacy"):
                    press_esc()
                    self.set_phase("DONE")
                    return

                if not trav.in_area("GOBLIN_VILLAGE") and not npc.closest_npc_by_name("General Bentnoze"):
                    trav.go_to("GOBLIN_VILLAGE", center=True)
                    return
                elif npc.closest_npc_by_name("General Wartface") and (not player.in_cutscene() or chat.can_continue()):
                    result = npc.chat_with_npc(
                        "General Wartface",
                        options=[
                            "I have some orange armour here.",
                            "I have some blue armour here.",
                            "I have some brown armour here.",
                            "Yes, he looks fat."
                        ]
                    )
                    if result is not None:
                        return result

            case "DONE":
                return


