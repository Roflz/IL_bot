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
from ..helpers.utils import press_esc


class GeTradePlan(Plan):
    id = "GE_TRADE"
    label = "Grand Exchange Trading"

    def __init__(self):
        self.state = {"phase": "GO_TO_GE"}
        self.next = self.state["phase"]
        self.loop_interval_ms = 600

    def compute_phase(self, payload, craft_recent):
        return self.state.get("phase", "GO_TO_GE")

    def set_phase(self, phase: str, ui=None, camera_setup: bool = True):
        from ..helpers.phase_utils import set_phase_with_camera
        return set_phase_with_camera(self, phase, ui, camera_setup)

    def loop(self, ui, payload):
        phase = self.state.get("phase", "GO_TO_GE")

        match(phase):
            case "GO_TO_GE":
                # Check if we're already at the Grand Exchange
                if trav.in_area("GE"):
                    self.set_phase("TRADE_PLAYER")
                    return
                else:
                    # Use enhanced long-distance travel for GE
                    print("[GE_TRADE] Using enhanced travel to reach Grand Exchange...")
                    result = trav.go_to("GE", use_long_distance=True)
                    return

            case "TRADE_PLAYER":
                # Look for a player to trade with
                print("[GE_TRADE] Looking for a player to trade with...")
                
                # Get all players in the area
                players = payload.get("players", [])
                if not players:
                    print("[GE_TRADE] No players found in area")
                    return
                
                # Find the closest player (excluding ourselves)
                my_name = payload.get("player", {}).get("name", "")
                closest_player = None
                closest_distance = float('inf')
                
                for player_data in players:
                    player_name = player_data.get("name", "")
                    if player_name == my_name or not player_name:
                        continue
                    
                    # Calculate distance (simple Manhattan distance)
                    player_world = player_data.get("world", {})
                    my_world = payload.get("player", {}).get("world", {})
                    
                    if (isinstance(player_world.get("x"), int) and isinstance(player_world.get("y"), int) and
                        isinstance(my_world.get("x"), int) and isinstance(my_world.get("y"), int)):
                        
                        distance = (abs(player_world["x"] - my_world["x"]) + 
                                  abs(player_world["y"] - my_world["y"]))
                        
                        if distance < closest_distance:
                            closest_distance = distance
                            closest_player = player_data
                
                if not closest_player:
                    print("[GE_TRADE] No suitable player found to trade with")
                    return
                
                player_name = closest_player.get("name", "Unknown")
                print(f"[GE_TRADE] Found player to trade with: {player_name} (distance: {closest_distance})")
                
                # Try to trade with the player
                print(f"[GE_TRADE] Attempting to trade with {player_name}...")
                
                # For now, just click on the player (you'll need to implement actual trading)
                result = npc.click_npc_simple(player_name)
                if result:
                    print(f"[GE_TRADE] Successfully clicked on {player_name}")
                    self.set_phase("DONE")
                else:
                    print(f"[GE_TRADE] Failed to click on {player_name}")

            case "DONE":
                print("[GE_TRADE] Trading complete!")
                return

        return None


def in_area(region_coords, payload):
    """Check if player is in a specific region"""
    try:
        player_world = payload.get("player", {}).get("world", {})
        if not player_world:
            return False
        
        px, py = player_world.get("x"), player_world.get("y")
        if not isinstance(px, int) or not isinstance(py, int):
            return False
        
        x1, y1, x2, y2 = region_coords
        return x1 <= px <= x2 and y1 <= py <= y2
    except Exception:
        return False
