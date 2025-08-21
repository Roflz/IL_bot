"""
Filter Utilities

Feature group filtering logic and helpers.
"""

from typing import List, Dict, Any, Callable
import numpy as np


def get_feature_group_filter_options() -> List[str]:
    """
    Get available feature group filter options.
    
    Returns:
        List of filter option strings
    """
    return [
        "All",
        "Player", 
        "Interaction",
        "Camera",
        "Inventory",
        "Bank",
        "Phase Context",
        "Game Objects",
        "NPCs",
        "Tabs",
        "Skills",
        "Timestamp"
    ]


def apply_feature_group_filter(feature_indices: List[int], 
                             feature_groups: Dict[int, str],
                             filter_group: str) -> List[int]:
    """
    Apply feature group filter to feature indices.
    
    Args:
        feature_indices: List of feature indices to filter
        feature_groups: Dictionary mapping feature index to group
        filter_group: Group to filter by ("All" for no filtering)
        
    Returns:
        Filtered list of feature indices
    """
    if filter_group == "All":
        return feature_indices
    
    filtered_indices = []
    for idx in feature_indices:
        if feature_groups.get(idx) == filter_group:
            filtered_indices.append(idx)
    
    return filtered_indices


def get_filtered_feature_data(sequence: np.ndarray,
                             feature_groups: Dict[int, str],
                             filter_group: str) -> tuple:
    """
    Get filtered feature data based on group filter.
    
    Args:
        sequence: Feature sequence array (10, 128)
        feature_groups: Dictionary mapping feature index to group
        filter_group: Group to filter by
        
    Returns:
        Tuple of (filtered_indices, filtered_data)
    """
    if filter_group == "All":
        # Return all features
        indices = list(range(128))
        data = sequence
    else:
        # Filter by group
        indices = []
        for i in range(128):
            if feature_groups.get(i) == filter_group:
                indices.append(i)
        
        if indices:
            data = sequence[:, indices]
        else:
            data = np.empty((10, 0))
    
    return indices, data


def create_feature_filter_callback(filter_var, feature_groups: Dict[int, str], 
                                 update_func: Callable) -> Callable:
    """
    Create a callback function for feature group filter changes.
    
    Args:
        filter_var: Tkinter variable for the filter
        feature_groups: Dictionary mapping feature index to group
        update_func: Function to call when filter changes
        
    Returns:
        Callback function
    """
    def on_filter_change(*args):
        """Handle filter change."""
        filter_group = filter_var.get()
        update_func(filter_group)
    
    return on_filter_change


def get_feature_group_stats(feature_groups: Dict[int, str]) -> Dict[str, int]:
    """
    Get statistics about feature groups.
    
    Args:
        feature_groups: Dictionary mapping feature index to group
        
    Returns:
        Dictionary mapping group names to counts
    """
    stats = {}
    for group in feature_groups.values():
        stats[group] = stats.get(group, 0) + 1
    return stats


def suggest_feature_filter(feature_groups: Dict[int, str], 
                         target_feature_idx: int) -> str:
    """
    Suggest a feature filter based on a target feature.
    
    Args:
        feature_groups: Dictionary mapping feature index to group
        target_feature_idx: Index of target feature
        
    Returns:
        Suggested filter group name
    """
    target_group = feature_groups.get(target_feature_idx, "other")
    
    # If target group has many features, suggest it
    group_stats = get_feature_group_stats(feature_groups)
    if target_group in group_stats and group_stats[target_group] > 5:
        return target_group
    
    # Otherwise suggest "All"
    return "All"
