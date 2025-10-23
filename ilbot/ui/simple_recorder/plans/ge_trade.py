# ge_trade.py
import logging
import time

from ..actions import objects, player, inventory
from ..actions.chat import find_chat_message, \
    click_chat_message
from ..actions.player import get_player_plane
from ..actions.timing import wait_until
from ..actions.trade import get_other_offer, other_offer_contains, accept_trade, other_offer_confirmation_contains, \
    accept_trade_confirm, get_players, find_player_by_name, trade_with_player, my_offer_contains, offer_all_items
from ..constants import BANK_REGIONS, REGIONS
import ilbot.ui.simple_recorder.actions.travel as trav
import ilbot.ui.simple_recorder.actions.bank as bank
import ilbot.ui.simple_recorder.actions.inventory as inv
import ilbot.ui.simple_recorder.actions.ge as ge
import ilbot.ui.simple_recorder.actions.npc as npc
import ilbot.ui.simple_recorder.actions.chat as chat

from .base import Plan
from .utilities.bank_plan import BankPlan
from ..helpers import quest
from ..helpers.bank import near_any_bank
from ..helpers.camera import move_camera_random
from ..helpers.utils import press_esc, press_spacebar
from ..helpers.widgets import widget_exists


class GeTradePlan(Plan):
    id = "GE_TRADE"
    label = "Grand Exchange Trading"

    def __init__(self, role="worker"):
        self.state = {"phase": "GO_TO_GE"}
        self.next = self.state["phase"]
        self.loop_interval_ms = 600
        self.role = role  # "worker" or "mule"

        # Create bank plan for withdrawing coins
        self.bank_plan = BankPlan(
            bank_area="CLOSEST_BANK",  # Use closest bank
            food_item=None,  # No food needed
            food_quantity=0,
            equipment_config={
                "weapon_tiers": [],  # No weapons needed
                "armor_tiers": {},   # No armor needed
                "jewelry_tiers": {}, # No jewelry needed
                "tool_tiers": []     # No tools needed
            },
            inventory_config={
                "required_items": [],  # No specific required items
                "optional_items": [],
                "deposit_all": True
            }
        )

        # Set up camera immediately during initialization
        from ilbot.ui.simple_recorder.helpers.camera import setup_camera_optimal
        setup_camera_optimal()

    def set_phase(self, phase: str, camera_setup: bool = True):
        from ..helpers.phase_utils import set_phase_with_camera
        return set_phase_with_camera(self, phase, camera_setup)

    def _handle_bank(self, ui) -> int:
        """Handle banking phase - withdraw all coins from bank."""
        # Update bank plan to withdraw all coins
        self.bank_plan.inventory_config["required_items"] = [
            {"name": "Coins", "quantity": -1}  # Withdraw all coins
        ]
        
        bank_status = self.bank_plan.loop(ui)
        
        if bank_status == BankPlan.SUCCESS:
            logging.info(f"[{self.id}] Banking completed successfully - coins withdrawn!")
            if self.role == "worker":
                self.set_phase("WORKER")
            else:  # mule
                self.set_phase("MULE")
            if bank.is_open():
                bank.close_bank()
            return self.loop_interval_ms
        
        elif bank_status == BankPlan.MISSING_ITEMS:
            error_msg = self.bank_plan.get_error_message()
            logging.warning(f"[{self.id}] No coins found in bank: {error_msg}")
            # Still proceed to GE even without coins
            if self.role == "worker":
                self.set_phase("DONE")
            else:  # mule
                self.set_phase("MULE")
            return self.loop_interval_ms
        
        elif bank_status == BankPlan.ERROR:
            error_msg = self.bank_plan.get_error_message()
            logging.error(f"[{self.id}] Banking error: {error_msg}")
            return self.loop_interval_ms
        
        else:
            # Still working on banking (TRAVELING, BANKING, etc.)
            return bank_status

    def loop(self, ui):
        phase = self.state.get("phase", "GO_TO_GE")
        logged_in = player.logged_in()
        if not logged_in:
            logging.info("Logged out, logging back in.")
            player.login()
            return self.loop_interval_ms

        match (phase):
            case "BANK":
                return self._handle_bank(ui)

            case "GO_TO_GE":
                # Check if we're already at the Grand Exchange
                if trav.in_area("GE"):
                    # Transition to role-based phase
                    self.set_phase("BANK")
                    return
                else:
                    # Use enhanced long-distance travel for GE
                    print(f"[GE_TRADE] Using enhanced travel to reach Grand Exchange as {self.role}...")
                    trav.go_to("GE")
                    return

            case "WORKER":
                if not widget_exists(21954562) and not widget_exists(21889025):
                    if not inventory.has_item("coins"):
                        self.set_phase("DONE")
                        return self.loop_interval_ms

                    if find_player_by_name("Batquinn"):
                        trade_with_player("Batquinn")
                        wait_until(lambda: widget_exists(21954562))
                        return self.loop_interval_ms

                elif widget_exists(21954562): #trade interface
                    if not my_offer_contains("coins"):
                        offer_all_items("coins")
                        return 2000
                    else:
                        accept_trade()
                        return 2000

                elif widget_exists(21889025):
                    accept_trade_confirm()
                    return 2000
                
                return self.loop_interval_ms

            case "MULE":
                # Random timer between 90 seconds and 4 minutes (240 seconds)
                import random
                timer_duration = random.randint(90, 240)
                logging.info(f"[GE_TRADE] Starting timer for {timer_duration} seconds")
                
                start_time = time.time()
                while time.time() - start_time < timer_duration:
                    if widget_exists(21954562): #trade interface
                        if other_offer_contains("coins"):
                            accept_trade()
                            return 2000
                        return
                    if widget_exists(21889025): #trade confirm interface
                        if other_offer_confirmation_contains("coins"):
                            accept_trade_confirm()
                            return 2000
                        return

                    message = find_chat_message("wishes to trade with you")
                    if message:
                        click_chat_message("wishes to trade with you")
                    
                    # Small delay to prevent excessive CPU usage
                    time.sleep(0.5)
                
                logging.info(f"[GE_TRADE] Timer completed after {timer_duration} seconds")
                logging.info("Moving camera a random amount")
                move_camera_random()
                return

            case "DONE":
                print("[GE_TRADE] Trading complete!")
                return

        return
