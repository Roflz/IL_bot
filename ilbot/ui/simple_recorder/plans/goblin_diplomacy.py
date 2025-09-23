# goblin_diplomacy.py
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
from ..helpers.nodes import add_node
from ..helpers.utils import press_enter, press_esc
from ..helpers.vars import get_var
from ..helpers.widgets import rect_center_from_widget


class GoblinDiplomacyPlan(Plan):
    id = "GOBLIN_DIPLOMACY"
    label = "Quest: Goblin Diplomacy"

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
                ui.debug(f"[GD] phase → {phase}")
            except Exception:
                pass
        return phase

    def loop(self, ui, payload):
        phase = self.state.get("phase", "GO_TO_CLOSEST_BANK")
        if quest.quest_finished("Goblin Diplomacy"):
            self.set_phase('DONE')

        match(phase):
            case "GO_TO_CLOSEST_BANK":
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
                    # Check what items we need (only unnoted items)
                    needed_items = []
                    required_items = ["Blue dye", "Orange dye", "Goblin mail"]
                    
                    for item in required_items:
                        # Check if we have the unnoted version
                        has_unnoted = inv.has_unnoted_item(item)
                        has_noted = inv.has_noted_item(item)
                        
                        if not has_unnoted:
                            if bank.has_item(item):
                                needed_items.append(item)
                            elif has_noted:
                                # We have noted version but need unnoted - need to withdraw unnoted
                                needed_items.append(item)
                    
                    if needed_items:
                        # Ensure bank note mode is disabled (withdraw as items, not notes)
                        bank.ensure_note_mode_disabled()
                        
                        # Deposit inventory to make space
                        bank.deposit_inventory()
                        wait_until(inv.is_empty, max_wait_ms=5000, min_wait_ms=200)
                        
                        # Withdraw needed items (as unnoted items)
                        for item in needed_items:
                            if item == "Goblin mail":
                                # Need 3 goblin mail
                                bank.withdraw_item(item, withdraw_x=3)
                            else:
                                bank.withdraw_item(item)
                        return
                    else:
                        # Check if we have all unnoted items in inventory
                        has_all_unnoted = all(inv.has_unnoted_item(item) for item in required_items)
                        if has_all_unnoted:
                            bank.close_bank()
                            if not wait_until(bank.is_closed, min_wait_ms=600, max_wait_ms=3000):
                                return
                            self.set_phase("MAKE_ARMOURS")
                            return
                        else:
                            # Missing items, need to buy from GE
                            bank.close_bank()
                            if not wait_until(bank.is_closed, min_wait_ms=600, max_wait_ms=3000):
                                return
                            self.set_phase("BUY_QUEST_ITEMS_FROM_GE")
                            return

            case "BUY_QUEST_ITEMS_FROM_GE":
                # Define required items
                required_items = ["Blue dye", "Orange dye", "Goblin mail"]
                
                # Check if we have all required items
                has_all_items = all(inv.has_item(item) for item in required_items)
                
                if has_all_items:
                    if ge.is_open():
                        ge.close_ge()
                    else:
                        self.set_phase("CHECK_BANK_FOR_QUEST_ITEMS")
                    return
                
                # Define items with quantities and price bumps
                items_to_buy = {
                    "Blue dye": (1, 5),
                    "Orange dye": (1, 5),
                    "Goblin mail": (3, 5)
                }
                
                # Use the centralized GE buying method
                ui.debug("[GD] Buying quest items from GE...")
                result = ge.buy_item_from_ge(items_to_buy, ui)
                if result is None:
                    # Still working on buying items, return to continue next tick
                    return
                elif result is True:
                    ui.debug("[GD] Successfully bought all quest items")
                else:
                    ui.debug("[GD] Failed to buy items, retrying...")
                    return
                
                # If we get here, all items should be bought
                if all(inv.has_item(item) for item in required_items):
                    self.set_phase("MAKE_ARMOURS")
                    return

            case "MAKE_ARMOURS":
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
                    self.set_phase("START_QUEST")
                    return

            case 'START_QUEST':
                if quest.quest_in_progress("Goblin Diplomacy"):
                    self.set_phase("TALK_TO_GENERAL_WARTFACE")
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
                # If we're carrying junk, clear it once so withdraw has space.
                if not inv.is_empty():
                    bank.deposit_inventory()
                    wait_until(inv.is_empty, max_wait_ms=4000, min_wait_ms=150)
                    return None

                # Withdraw if present; otherwise move on.
                if bank.has_item(item):
                    if item == "Goblin mail":
                        bank.withdraw_item(item, quantity=3)
                    else:
                        bank.withdraw_item(item)
                    wait_until(lambda: inv.has_item(item), max_wait_ms=4000)
                    bank.close_bank()
                    return None  # next loop sees inv.has_item(item) and returns True
                else:
                    bank.close_bank()
                    # fall through to GE
                    # (don't return True—still need to buy)

        # 3) Not in bank, buy at GE (this function is fully idempotent across ticks)
        # Special quantity handling for Goblin mail
        quantity = 3 if item == "Goblin mail" else 1
        return ge.buy_item_from_ge({item: (quantity, 5)}, ui)

