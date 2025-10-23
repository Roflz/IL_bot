"""
Login utility plan - handles logging into the game.
"""

import logging
from ..base import Plan
from ...actions import player


class LoginPlan(Plan):
    """Simple login utility plan."""
    
    id = "login"
    label = "Login Utility"
    
    def __init__(self):
        self.state = {"phase": "LOGIN"}
        self.next = self.state["phase"]
        self.loop_interval_ms = 1000
    
    def loop(self, ui) -> int:
        """Main loop method."""
        phase = self.state.get("phase", "LOGIN")
        
        if phase == "LOGIN":
            logged_in = player.logged_in()
            if not logged_in:
                player.login()
                return self.loop_interval_ms
            else:
                self.set_phase("DONE")
                return self.loop_interval_ms
        elif phase == "DONE":
            return self.loop_interval_ms
        else:
            return self.loop_interval_ms