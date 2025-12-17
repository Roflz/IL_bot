# ge_trade.py
import logging

from actions import player
import actions.travel as trav
import actions.bank as bank

from .base import Plan
from helpers import move_camera_random, setup_camera_optimal
from helpers.utils import sleep_exponential
from helpers import set_phase_with_camera


class GeTradePlan(Plan):
    id = "GE_TRADE"
    label = "Grand Exchange Trading"

    def __init__(self):
        self.state = {"phase": "GO_TO_GE"}
        self.next = self.state["phase"]
        self.loop_interval_ms = 600

        # Set up camera immediately during initialization
        setup_camera_optimal()

    def set_phase(self, phase: str, camera_setup: bool = True):
        return set_phase_with_camera(self, phase, camera_setup)

    def loop(self, ui):
        phase = self.state.get("phase", "GO_TO_GE")
        logged_in = player.logged_in()
        if not logged_in:
            logging.info("Logged out, logging back in.")
            player.login()
            return self.loop_interval_ms

        match (phase):
            case "GO_TO_GE":
                # Check if we're already at the Grand Exchange
                if trav.in_area("GE"):
                    self.set_phase("REMOVE_EQUIPMENT", ui)
                    return
                else:
                    # Use enhanced long-distance travel for GE
                    print("[GE_TRADE] Using enhanced travel to reach Grand Exchange...")
                    result = trav.go_to("GE")
                    return

            case "REMOVE_EQUIPMENT":
                bank.open_bank()
                bank.deposit_equipment()
                bank.close_bank()
                self.set_phase("WAIT", ui)

            case "WAIT":
                logging.info("Moving camera a random amount")
                move_camera_random()
                sleep_exponential(50, 70, 1.0)
                return

            case "DONE":
                print("[GE_TRADE] Trading complete!")
                return

        return
