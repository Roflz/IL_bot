# ge_trade.py
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
from ..helpers.utils import press_esc, press_spacebar


class GeTradePlan(Plan):
    id = "GE_TRADE"
    label = "Grand Exchange Trading"

    def __init__(self):
        self.state = {"phase": "GO_TO_GE"}
        self.next = self.state["phase"]
        self.loop_interval_ms = 600
        
        # Set up camera immediately during initialization
        from ilbot.ui.simple_recorder.helpers.camera import setup_camera_optimal
        setup_camera_optimal()


    def set_phase(self, phase: str, camera_setup: bool = True):
        from ..helpers.phase_utils import set_phase_with_camera
        return set_phase_with_camera(self, phase, ui, camera_setup)

    def loop(self, ui):
        phase = self.state.get("phase", "GO_TO_GE")

        match(phase):
            case "GO_TO_GE":
                # Check if we're already at the Grand Exchange
                if trav.in_area("GE"):
                    self.set_phase("TRADE_PLAYER", ui)
                    return
                else:
                    # Use enhanced long-distance travel for GE
                    print("[GE_TRADE] Using enhanced travel to reach Grand Exchange...")
                    result = trav.go_to("GE")
                    return

            case "TRADE_PLAYER":
                time.sleep(300)
                print("pressing space bar")
                press_spacebar()
                return

            case "DONE":
                print("[GE_TRADE] Trading complete!")
                return

        return
