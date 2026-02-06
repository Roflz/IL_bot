"""
Wait utility plan - waits for a specified amount of time.
"""

import logging
import time
from ..base import Plan


class WaitPlan(Plan):
    """Simple wait utility plan."""
    
    id = "wait"
    label = "Wait Utility"
    description = """Simple utility plan that waits for a specified amount of time (in minutes). Useful for adding delays between plans or creating timed sequences. Configurable wait duration via parameters.

Starting Area: Anywhere
Required Items: None"""
    
    def __init__(self, wait_minutes=1, **kwargs):
        self.state = {"phase": "WAIT"}
        self.next = self.state["phase"]
        self.loop_interval_ms = 1000
        
        # Get wait_minutes from parameters or use default
        self.wait_minutes = float(kwargs.get('wait_minutes', wait_minutes))
        self.wait_seconds = self.wait_minutes * 60  # Convert minutes to seconds
        self.start_time = None

    def set_phase(self, phase: str, camera_setup: bool = True):
        self.state["phase"] = phase
        self.next = phase
        return phase

    
    def loop(self, ui) -> int:
        """Main loop method."""
        phase = self.state.get("phase", "WAIT")
        
        if phase == "WAIT":
            if self.start_time is None:
                self.start_time = time.time()
                wait_minutes = self.wait_seconds / 60
                logging.info(f"[{self.id}] Starting wait for {wait_minutes:.1f} minutes")
            
            elapsed = time.time() - self.start_time
            remaining = self.wait_seconds - elapsed
            
            if remaining <= 0:
                elapsed_minutes = elapsed / 60
                logging.info(f"[{self.id}] Wait completed after {elapsed_minutes:.1f} minutes")
                self.set_phase("DONE")
            else:
                remaining_minutes = remaining / 60
                logging.info(f"[{self.id}] Waiting... {remaining_minutes:.1f} minutes remaining")
            
            return self.loop_interval_ms
            
        elif phase == "DONE":
            return self.loop_interval_ms
        else:
            return self.loop_interval_ms
