# romeo_and_juliet_loop.py (your immediate-mode plan)
import time

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
from ..helpers.ge import widget_by_id_text, ge_buy_confirm_widget
from ..helpers.widgets import rect_center_from_widget


class RomeoAndJulietPlan(Plan):
    id = "ROMEO_AND_JULIET"
    label = "Quest: Romeo & Juliet"


    def __init__(self):
        self.state = {"phase": "GO_TO_CLOSEST_BANK"}
        self.next = self.state["phase"]
        self.loop_interval_ms = 600

    def compute_phase(self, payload, craft_recent):
        return self.state.get("phase", "GO_TO_CLOSEST_BANK")

    def set_phase(self, phase: str, ui=None):
        self.state["phase"] = phase
        self.next = phase
        if ui is not None:
            try:
                ui.debug(f"[RJ] phase → {phase}")
            except Exception:
                pass
        return phase

    def loop(self, ui, payload):
        # phase = self.state.get("phase", "GO_TO_CLOSEST_BANK")
        phase = "TALK_TO_JULIET_1"

        match(phase):
            case "GO_TO_CLOSEST_BANK":
                trav.go_to_closest_bank(payload)
                ui.debug("[RJ] Reached closest bank; advancing to DONE")
                self.state["phase"] = "CHECK_BANK_FOR_QUEST_ITEMS"
                return

            case "CHECK_BANK_FOR_QUEST_ITEMS":
                if bank.is_closed():
                    bank.open_bank()
                    wait_until(bank.is_open, max_wait_ms=5000, min_wait_ms=1000)
                    bank.deposit_inventory()
                    wait_until(inv.is_empty)
                    if bank.has_item("Cavada berries"):
                        bank.withdraw_item("Cavada berries")
                        bank.close_bank()
                        self.set_phase("DONE")
                    else:
                        bank.close_bank()
                        self.set_phase("BUY_QUEST_ITEMS_FROM_GE")
                else:
                    bank.close_bank()

            case "BUY_QUEST_ITEMS_FROM_GE":
                if inv.has_item("Cadava berries") and ge.is_closed():
                    self.set_phase("START_QUEST")
                # Go to GE if not there
                if not trav.in_area(BANK_REGIONS["GE"]):
                    trav.go_to_ge()
                # Once you are in the GE:
                else:
                    if inv.has_item("coins") and ge.is_closed() and bank.is_closed():
                        ge.open_ge()
                        wait_until(ge.is_open, max_wait_ms=5000)
                        return
                    elif inv.has_item("coins") and ge.is_open():
                        ge.begin_buy_offer()
                        wait_until(ge.ge_offer_open, max_wait_ms=5000)
                        ge.type_item_name("Cadava berries")
                        wait_until(lambda: ge.selected_item_is("Cadava berries"))
                        for _ in range(5):
                            plus = widget_by_id_text( 30474266, "+5%")
                            if not plus:
                                break
                            cx, cy = rect_center_from_widget(plus)
                            ui.dispatch({
                                "id": "ge-plus5",
                                "action": "click",
                                "description": "+5% price",
                                "target": {"name": "+5%", "bounds": plus.get("bounds")},
                                "click": {"type": "point", "x": cx, "y": cy},
                            })
                            time.sleep(0.3)
                        ge.confirm_buy()
                        wait_until(lambda: ge_buy_confirm_widget(payload) is None, max_wait_ms=8000)
                        ge.collect_to_inventory()
                        wait_until(lambda: inv.has_item("Cadava berries"))
                        ge.close_ge()

                        return
                    elif not inv.has_item("coins") and bank.is_closed():
                        bank.open_bank()
                        wait_until(bank.is_open, max_wait_ms=6000)
                        bank.deposit_inventory()
                        bank.withdraw_item("coins", withdraw_all=True)
                        wait_until(lambda: inv.has_item("coins"))
                        return
                    elif bank.is_open():
                        bank.close_bank()
                        return

            case 'START_QUEST':
                if quest.quest_in_progress("Romeo & Juliet"):
                    self.set_phase("TALK_TO_JULIET_1")
                if not trav.in_area(REGIONS["VARROCK_SQUARE"]):
                    trav.go_to("VARROCK_SQUARE")
                    return
                elif npc.closest_npc_by_name("Romeo"):
                    if not chat.dialogue_is_open() and not chat.can_choose_option():
                        npc.click_npc("Romeo")
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
                if not trav.in_area(REGIONS["JULIET_MANSION"]):
                    trav.go_to("JULIET_MANSION")
                    return



            case "DONE":
                return
