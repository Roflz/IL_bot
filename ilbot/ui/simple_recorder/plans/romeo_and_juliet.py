# romeo_and_juliet_loop.py (your immediate-mode plan)
import time

from ..actions import objects, player
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
from ..helpers.npc import closest_npc_by_name
from ..helpers.utils import press_esc


class RomeoAndJulietPlan(Plan):
    id = "ROMEO_AND_JULIET"
    label = "Quest: Romeo & Juliet"

    def __init__(self):
        self.state = {"phase": "GO_TO_CLOSEST_BANK"}  # gate: ensure items first
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

        match(phase):
            case "GO_TO_CLOSEST_BANK":
                if inv.has_item("Cadava Berries") and quest.quest_state("Romeo & Juliet") == 'NOT_STARTED':
                    self.set_phase("START_QUEST", ui)
                    return
                if not near_any_bank():
                    trav.go_to_closest_bank()
                else:
                    self.set_phase("CHECK_BANK_FOR_QUEST_ITEMS", ui)
                return

            case "CHECK_BANK_FOR_QUEST_ITEMS":
                if bank.is_closed():
                    bank.open_bank()
                    return
                elif bank.is_open():
                    if inv.has_item("Cadava berries"):
                        bank.close_bank()
                        if not wait_until(bank.is_closed, min_wait_ms=600, max_wait_ms=3000):
                            return
                        self.set_phase("START_QUEST", ui)
                        return
                    elif bank.has_item("Cadava berries"):
                        bank.deposit_inventory()
                        wait_until(inv.is_empty, max_wait_ms=5000, min_wait_ms=200)
                        bank.withdraw_item("Cadava berries")
                        return
                    else:
                        bank.close_bank()
                        if not wait_until(bank.is_closed, min_wait_ms=600, max_wait_ms=3000):
                            return
                        self.set_phase("BUY_QUEST_ITEMS_FROM_GE", ui)
                        return

            case "BUY_QUEST_ITEMS_FROM_GE":
                if inv.has_item("Cadava berries") and ge.is_closed():
                    self.set_phase("START_QUEST", ui)
                    return
                
                # Use the centralized GE buying method
                result = ge.buy_item_from_ge("Cadava berries", ui)
                if result is True:
                    self.set_phase("START_QUEST", ui)
                    return

            case 'START_QUEST':
                if quest.quest_in_progress("Romeo & Juliet"):
                    self.set_phase("TALK_TO_JULIET_1", ui)
                    return
                if not trav.in_area(REGIONS["VARROCK_SQUARE"]) and not closest_npc_by_name("Romeo"):
                    trav.go_to_and_find_npc("VARROCK_SQUARE", "Romeo")
                    return
                elif npc.closest_npc_by_name("Romeo"):
                    if not chat.dialogue_is_open() and not chat.can_choose_option():
                        npc.click_npc_action("Romeo", "Talk-to")
                        wait_until(chat.dialogue_is_open, max_wait_ms=4000)
                        return
                    else:
                        if chat.dialogue_is_open():
                            chat.continue_dialogue()
                            return
                        if chat.can_choose_option():
                            if chat.option_exists("Perhaps I could help"):
                                chat.choose_option("Perhaps I could help")
                                return 1200
                            elif chat.option_exists("Yes."):
                                chat.choose_option("Yes.")
                                return 1200

            case "TALK_TO_JULIET_1":
                if inv.has_item("Message"):
                    self.set_phase("TALK_TO_ROMEO_1", ui)
                    return
                elif not trav.in_area(REGIONS["JULIET_MANSION"]) and get_player_plane() == 0:
                    trav.go_to("JULIET_MANSION")
                    return
                elif get_player_plane() == 0:
                    objects.click_object_action("Staircase", "Climb-up")
                    wait_until(lambda: get_player_plane() == 1, max_wait_ms=5000)
                    return
                elif get_player_plane() == 1 and not chat.dialogue_is_open():
                    npc.click_npc_action("Juliet", "Talk-to")
                    return
                elif chat.dialogue_is_open():
                    chat.continue_dialogue()
                    return

            case "TALK_TO_ROMEO_1":
                if chat.dialogue_contains("Oh yes, Father Lawrence...") or not inv.has_item("Message"):
                    chat.continue_dialogue()
                    self.set_phase("FATHER_LAWRENCE", ui)
                    return
                if get_player_plane() == 1:
                    objects.click_object_action("Staircase", "Climb-down")
                    wait_until(lambda: get_player_plane() == 0, max_wait_ms=5000)
                    return 2000
                elif not trav.in_area(REGIONS["VARROCK_SQUARE"]):
                    trav.go_to_and_find_npc("VARROCK_SQUARE", "Romeo")
                    return
                elif not chat.dialogue_is_open() and not chat.can_continue():
                    npc.click_npc_action("Romeo", "Talk-to")
                    return 3000
                else:
                    chat.continue_dialogue()
                    return

            case "FATHER_LAWRENCE":
                if chat.dialogue_contains("Apart from the strong overtones") or chat.dialogue_contains("Ah, have you found the Apothecary yet?"):
                    chat.continue_dialogue()
                    self.set_phase("GET_POTION", ui)
                    return
                if not trav.in_area(REGIONS["VARROCK_CHURCH"]):
                    trav.go_to("VARROCK_CHURCH")
                    return
                elif not chat.dialogue_is_open() and not chat.can_continue() and not player.in_cutscene():
                    npc.click_npc_action("Father Lawrence", "Talk-to")
                    return 3000
                else:
                    chat.continue_dialogue()
                    return

            case "GET_POTION":
                if inv.has_item("Cadava potion"):
                    self.set_phase("GIVE_POTION_TO_JULIET", ui)
                    return
                if closest_npc_by_name("Apothecary"):
                    if not chat.dialogue_is_open() and not chat.can_continue() and not chat.get_options():
                        npc.click_npc_action("Apothecary", "Talk-to")
                        return 3000
                    elif chat.can_choose_option():
                        if chat.option_exists("Talk about something else."):
                            chat.choose_option("Talk about something else.")
                            return 1200
                        elif chat.option_exists("Talk about Romeo & Juliet."):
                            chat.choose_option("Talk about Romeo & Juliet.")
                            return 1200
                    else:
                        chat.continue_dialogue()
                        return
                elif not trav.in_area(REGIONS["VARROCK_APOTHECARY"]):
                    trav.go_to("VARROCK_APOTHECARY")
                    return

            case "GIVE_POTION_TO_JULIET":
                if not inv.has_item("Cadava potion"):
                    self.set_phase("FINISH_QUEST", ui)
                    return
                if not trav.in_area(REGIONS["JULIET_MANSION"]) and get_player_plane() == 0:
                    trav.go_to("JULIET_MANSION")
                    return
                elif get_player_plane() == 0:
                    objects.click_object_action("Staircase", "Climb-up")
                    wait_until(lambda: get_player_plane() == 1, max_wait_ms=5000)
                    return
                elif get_player_plane() == 1 and not chat.can_continue() and not player.in_cutscene():
                    npc.click_npc_action("Juliet", "Talk-to")
                    return
                elif chat.can_continue() and not chat.dialogue_contains("Please go to Romeo and make sure he understands."):
                    chat.continue_dialogue()
                    return
                elif chat.can_continue():
                    chat.continue_dialogue()
                    wait_until(player.in_cutscene, max_wait_ms=3000)
                    return
                else:
                    return

            case "FINISH_QUEST":
                if quest.quest_state("Romeo & Juliet") == "FINISHED":
                    self.set_phase("DONE", ui)
                    press_esc()
                    return
                if get_player_plane() == 1 and not player.in_cutscene():
                    objects.click_object_action("Staircase", "Climb-down")
                    wait_until(lambda: get_player_plane() == 0, max_wait_ms=5000)
                    return
                elif not trav.in_area(REGIONS["VARROCK_SQUARE"]) and not player.in_cutscene():
                    trav.go_to_and_find_npc("VARROCK_SQUARE", "Romeo")
                    return
                elif player.in_cutscene():
                    chat.continue_dialogue()
                    return
                elif not chat.dialogue_is_open() and not chat.can_continue() and not player.in_cutscene():
                    npc.click_npc_action("Romeo", "Talk-to")
                    return 3000
                elif chat.dialogue_is_open():
                    chat.continue_dialogue()
                    return 3000

            case "DONE":
                return


