"""
UI State Management

Contains the dataclass for UI state and related constants.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class UIState:
    """UI state container for the training browser."""
    
    # Data and navigation
    data_root: str = "data"
    current_sequence: int = 0
    
    # Display toggles
    show_translations: bool = True
    show_normalized: bool = False
    
    # Filters
    feature_group_filter: str = "All"
    
    # Action display settings
    show_action_normalized: bool = False
    
    # Validation
    def __post_init__(self):
        """Validate state after initialization."""
        if self.current_sequence < 0:
            self.current_sequence = 0
        if self.feature_group_filter not in ["All", "Player", "Interaction", "Camera", 
                                           "Inventory", "Bank", "Phase Context", 
                                           "Game Objects", "NPCs", "Tabs", "Skills", "Timestamp"]:
            self.feature_group_filter = "All"
