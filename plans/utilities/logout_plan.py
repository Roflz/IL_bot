"""
Logout utility plan - handles logging out of the game.
"""

from ..base import Plan
from actions import player


class LogoutPlan(Plan):
    """Simple logout utility plan."""
    
    id = "logout"
    label = "Logout Utility"
    
    def __init__(self):
        self.state = {"phase": "LOGOUT"}
        self.next = self.state["phase"]
        self.loop_interval_ms = 1000
    
    def loop(self, ui) -> int:
        """Main loop method."""
        phase = self.state.get("phase", "LOGOUT")
        
        if phase == "LOGOUT":
            logged_in = player.logged_in()
            if logged_in:
                success = player.logout()
                if success:
                    self.set_phase("DONE")
                return self.loop_interval_ms
            else:
                self.set_phase("DONE")
                return self.loop_interval_ms
        elif phase == "DONE":
            return self.loop_interval_ms
        else:
            return self.loop_interval_ms
