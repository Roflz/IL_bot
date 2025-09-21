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
from ..helpers.ge import widget_by_id_text, ge_buy_confirm_widget
from ..helpers.utils import press_enter
from ..helpers.vars import get_var
from ..helpers.widgets import rect_center_from_widget


class RomeoAndJulietPlan(Plan):
    id = "ROMEO_AND_JULIET"
    label = "Quest: Romeo & Juliet"

    def __init__(self):
        self.state = {"phase": "GO_TO_CLOSEST_BANK"}  # gate: ensure items first
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
        phase = self.state.get("phase", "GO_TO_CLOSEST_BANK")
        # phase = "FINISH_QUEST"
        # if phase == "GO_TO_CLOSEST_BANK":
        #     self.set_phase("FATHER_LAWRENCE")

        match(phase):
            case "GO_TO_CLOSEST_BANK":
                if inv.has_item("Cadava Berries") and quest.quest_state("Romeo & Juliet") == 'NOT_STARTED':
                    self.set_phase("START_QUEST")
                    return
                if not near_any_bank(payload):
                    trav.go_to_closest_bank(payload)
                else:
                    self.state["phase"] = "CHECK_BANK_FOR_QUEST_ITEMS"
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
                        self.set_phase("START_QUEST")
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
                        self.set_phase("BUY_QUEST_ITEMS_FROM_GE")
                        return

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
                        if not ge.offer_open():
                            ge.begin_buy_offer()
                            if not wait_until(ge.ge_offer_open, max_wait_ms=5000):
                                return
                        ge.type_item_name("Cadava berries")
                        if wait_until(lambda: ge.buy_chatbox_first_item_is("Cadava berries")):
                            press_enter()
                            return
                        if not wait_until(lambda: ge.selected_item_is("Cadava berries")):
                            return
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
                        if not wait_until(bank.is_open, max_wait_ms=6000):
                            return
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
                if inv.has_item("Message"):
                    self.set_phase("TALK_TO_ROMEO_1")
                    return
                elif not trav.in_area(REGIONS["JULIET_MANSION"]) and get_player_plane() == 0:
                    trav.go_to("JULIET_MANSION")
                elif get_player_plane() == 0:
                    objects.click("Staircase")
                elif get_player_plane() == 1 and not chat.dialogue_is_open():
                    npc.click_npc("Juliet")
                elif chat.dialogue_is_open():
                    chat.continue_dialogue()
                    return

            case "TALK_TO_ROMEO_1":
                if chat.dialogue_contains("Oh yes, Father Lawrence..."):
                    chat.continue_dialogue()
                    self.set_phase("FATHER_LAWRENCE")
                if get_player_plane() == 1:
                    objects.click("Staircase")
                    return
                elif not trav.in_area(REGIONS["VARROCK_SQUARE"]):
                    trav.go_to("VARROCK_SQUARE")
                    return
                elif not chat.dialogue_is_open() and not chat.can_continue():
                    npc.click_npc("Romeo")
                    return 3000
                else:
                    chat.continue_dialogue()
                    return

            case "FATHER_LAWRENCE":
                if chat.dialogue_contains("Apart from the strong overtones") or chat.dialogue_contains("Ah, have you found the Apothecary yet?"):
                    chat.continue_dialogue()
                    self.set_phase("GET_POTION")
                    return
                if not trav.in_area(REGIONS["VARROCK_CHURCH"]):
                    trav.go_to("VARROCK_CHURCH")
                    return
                elif not chat.dialogue_is_open() and not chat.can_continue() and not player.in_cutscene():
                    npc.click_npc("Father Lawrence")
                    return 3000
                else:
                    chat.continue_dialogue()
                    return

            case "GET_POTION":
                if inv.has_item("Cadava potion"):
                    self.set_phase("GIVE_POTION_TO_JULIET")
                    return
                elif not trav.in_area(REGIONS["VARROCK_APOTHECARY"]):
                    trav.go_to("VARROCK_APOTHECARY")
                    return
                elif not chat.dialogue_is_open() and not chat.can_continue() and not chat.get_options():
                    npc.click_npc("Apothecary")
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

            case "GIVE_POTION_TO_JULIET":
                if not inv.has_item("Cadava potion"):
                    self.set_phase("FINISH_QUEST")
                if not trav.in_area(REGIONS["JULIET_MANSION"]) and get_player_plane() == 0:
                    trav.go_to("JULIET_MANSION")
                    return
                elif get_player_plane() == 0:
                    objects.click("Staircase")
                    return
                elif get_player_plane() == 1 and not chat.can_continue() and not player.in_cutscene():
                    npc.click_npc("Juliet")
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
                    self.set_phase("DONE")
                    return
                if get_player_plane() == 1 and not player.in_cutscene():
                    objects.click("Staircase")
                    return
                elif not trav.in_area(REGIONS["VARROCK_SQUARE"]) and not player.in_cutscene():
                    trav.go_to("VARROCK_SQUARE")
                    return
                elif not chat.dialogue_is_open() and not chat.can_continue() and not player.in_cutscene():
                    npc.click_npc("Romeo")
                    return 3000
                elif chat.dialogue_is_open():
                    chat.continue_dialogue()
                    return



            case "DONE":
                return

    def ensure_have_item(self, item: str, ui, payload) -> bool | None:
        """
        Idempotent: returns True only when the item is in inventory.
        Otherwise it performs the next minimal step toward getting it and returns None.
        """
        # 1) Already in inventory?
        if inv.has_item(item):
            return True

        # 2) Nearby bank? Open and try to withdraw.
        if near_any_bank(payload):
            if bank.is_closed():
                bank.open_bank()
                wait_until(bank.is_open, max_wait_ms=6000)
                return None

            if bank.is_open():
                # If we’re carrying junk, clear it once so withdraw has space.
                if not inv.is_empty():
                    bank.deposit_inventory()
                    wait_until(inv.is_empty, max_wait_ms=4000, min_wait_ms=150)
                    return None

                # Withdraw if present; otherwise move on.
                if bank.has_item(item):
                    bank.withdraw_item(item)
                    wait_until(lambda: inv.has_item(item), max_wait_ms=4000)
                    bank.close_bank()
                    return None  # next loop sees inv.has_item(item) and returns True
                else:
                    bank.close_bank()
                    # fall through to GE
                    # (don’t return True—still need to buy)

        # 3) Not in bank, buy at GE (this function is fully idempotent across ticks)
        return self._buy_item_from_ge(item, ui, payload)

    def _buy_item_from_ge(self, item: str, ui, payload, price_bumps: int = 5) -> bool | None:
        """
        Returns True when item is in inventory.
        Otherwise performs exactly one small step toward buying it and returns None.
        Safe to call every tick.
        """
        # If we already have it (race), bail out True.
        if inv.has_item(item):
            return True

        # Go to GE
        if not trav.in_area(BANK_REGIONS["GE"]):
            trav.go_to_ge()
            return None

        # Ensure coins: if we don’t have any, grab from bank quickly (GE has a bank close by)
        if not inv.has_item("coins"):
            if bank.is_closed():
                bank.open_bank()
                wait_until(bank.is_open, max_wait_ms=6000)
                return None
            if bank.is_open():
                # make space then withdraw all coins
                if not inv.is_empty():
                    bank.deposit_inventory()
                    wait_until(inv.is_empty, max_wait_ms=4000, min_wait_ms=150)
                    return None
                bank.withdraw_item("coins", withdraw_all=True)
                wait_until(lambda: inv.has_item("coins"), max_wait_ms=3000)
                return None

        # Open GE
        if ge.is_closed():
            ge.open_ge()
            wait_until(ge.is_open, max_wait_ms=5000)
            return None

        # Open buy offer panel
        if not ge.offer_open():
            ge.begin_buy_offer()
            if not wait_until(ge.ge_offer_open, max_wait_ms=5000):
                return None
            return None

        # Type item name and select first result
        ge.type_item_name(item)
        if wait_until(lambda: ge.buy_chatbox_first_item_is(item), max_wait_ms=2000):
            # You added a direct key helper; use it to confirm the selection fast.
            press_enter()
            return None

        # Wait until the selected item is locked in
        if not wait_until(lambda: ge.selected_item_is(item), max_wait_ms=2000):
            return None

        # Nudge price a few times (+5%)
        for _ in range(max(0, int(price_bumps))):
            plus = widget_by_id_text(30474266, "+5%")
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
            time.sleep(0.25)

        # Confirm buy
        ge.confirm_buy()
        wait_until(lambda: ge_buy_confirm_widget(payload) is None, max_wait_ms=8000)

        # Collect to inventory
        ge.collect_to_inventory()
        if not wait_until(lambda: inv.has_item(item), max_wait_ms=4000):
            return None

        ge.close_ge()
        return True
