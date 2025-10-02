"""
Utility functions for plan phase management with camera setup.

Usage in your plan classes:

    def set_phase(self, phase: str, ui=None, camera_setup: bool = True):
        from ..helpers.phase_utils import set_phase_with_camera
        return set_phase_with_camera(self, phase, ui, camera_setup)

This will automatically:
- Set the new phase in self.state
- Set self.next to the new phase
- Run camera setup for new phases (unless camera_setup=False)
- Log the phase change to UI debug output

The camera setup uses setup_camera_optimal() which:
- Sets optimal zoom level (scale 551)
- Sets optimal pitch angle (383 degrees)
- Provides better view for bot operations
"""
from typing import Optional, Dict, Any


def set_phase_with_camera(self, phase: str, ui=None, camera_setup: bool = True) -> str:
    """
    Set a new phase and optionally set up camera for the new phase.
    
    Args:
        self: The plan instance (should have state attribute)
        phase: The new phase to set
        ui: Optional UI instance for debugging
        camera_setup: Whether to run camera setup for new phases (default True)
        
    Returns:
        The phase that was set
    """
    # Check if this is a new phase (not just setting the same phase)
    current_phase = self.state.get("phase")
    is_new_phase = current_phase != phase
    
    self.state["phase"] = phase
    self.next = phase
    
    # Set up camera for new phases
    if is_new_phase and camera_setup:
        try:
            from ilbot.ui.simple_recorder.helpers.camera import setup_camera_optimal
            from ilbot.ui.simple_recorder.helpers.context import get_payload
            print(f"[{self.__class__.__name__}] Setting up camera for new phase: {phase}")
            setup_camera_optimal(get_payload())
        except Exception as e:
            print(f"[{self.__class__.__name__}] Camera setup failed for phase {phase}: {e}")
    
    if ui is not None:
        try:
            ui.debug(f"[{self.__class__.__name__}] phase → {phase}")
        except Exception:
            pass
    return phase


def create_phase_manager(plan_class_name: str):
    """
    Create a phase manager function for a specific plan class.
    
    Args:
        plan_class_name: The name of the plan class (for logging)
        
    Returns:
        A function that can be used as a set_phase method
    """
    def set_phase(self, phase: str, ui=None, camera_setup: bool = True) -> str:
        return set_phase_with_camera(self, phase, ui, camera_setup)
    
    return set_phase
