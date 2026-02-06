"""
Login utility plan - handles logging into the game.
"""

from ..base import Plan
from actions import player


class LoginPlan(Plan):
    """Simple login utility plan."""
    
    id = "login"
    label = "Login Utility"
    description = """Simple utility plan that handles logging into the game. Useful as a first step in plan sequences to ensure the character is logged in before other operations.

Starting Area: Login screen
Required Items: None"""
    
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