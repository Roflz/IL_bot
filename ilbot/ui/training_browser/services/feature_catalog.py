"""
Feature Catalog Service

Wraps feature_map helpers for index→name/group/type operations.
"""

from typing import Dict, List, Optional, Any
from ilbot.pipeline.shared_pipeline.feature_map import (
    get_feature_group_for_index, get_feature_info, validate_feature_mappings
)


class FeatureCatalog:
    """Service for feature mapping operations."""
    
    def __init__(self, feature_mappings: List[Dict[str, Any]]):
        """
        Initialize the feature catalog.
        
        Args:
            feature_mappings: List of feature mapping dictionaries
        """
        self.feature_mappings = feature_mappings
        self._validate_mappings()
        self._build_index_maps()
    
    def _validate_mappings(self):
        """Validate feature mappings."""
        try:
            validate_feature_mappings(self.feature_mappings)
        except Exception as e:
            print(f"⚠ Warning: Feature mappings validation failed: {e}")
    
    def _build_index_maps(self):
        """Build efficient index-based lookup maps."""
        self.feature_names = {}
        self.feature_groups = {}
        self.feature_types = {}
        
        for mapping in self.feature_mappings:
            feature_idx = mapping.get('feature_index')
            if feature_idx is not None:
                self.feature_names[feature_idx] = mapping.get('feature_name', f'feature_{feature_idx}')
                self.feature_groups[feature_idx] = mapping.get('feature_group', 'other')
                self.feature_types[feature_idx] = mapping.get('data_type', 'unknown')
    
    def get_feature_name(self, feature_idx: int) -> str:
        """
        Get feature name for a given index.
        
        Args:
            feature_idx: Feature index (0-127)
            
        Returns:
            Feature name string
        """
        return self.feature_names.get(feature_idx, f'feature_{feature_idx}')
    
    def get_feature_group(self, feature_idx: int) -> str:
        """
        Get feature group for a given index.
        
        Args:
            feature_idx: Feature index (0-127)
            
        Returns:
            Feature group string
        """
        return self.feature_groups.get(feature_idx, 'other')
    
    def get_feature_type(self, feature_idx: int) -> str:
        """
        Get feature data type for a given index.
        
        Args:
            feature_idx: Feature index (0-127)
            
        Returns:
            Feature data type string
        """
        return self.feature_types.get(feature_idx, 'unknown')
    
    def get_feature_info(self, feature_idx: int) -> Optional[Dict[str, Any]]:
        """
        Get complete feature info for a given index.
        
        Args:
            feature_idx: Feature index (0-127)
            
        Returns:
            Feature info dictionary or None if not found
        """
        try:
            return get_feature_info(self.feature_mappings, feature_idx)
        except Exception:
            return None
    
    def get_features_by_group(self, group_name: str) -> List[int]:
        """
        Get all feature indices for a given group.
        
        Args:
            group_name: Feature group name
            
        Returns:
            List of feature indices
        """
        return [idx for idx, group in self.feature_groups.items() if group == group_name]
    
    def get_features_by_type(self, data_type: str) -> List[int]:
        """
        Get all feature indices for a given data type.
        
        Args:
            data_type: Feature data type
            
        Returns:
            List of feature indices
        """
        return [idx for idx, ftype in self.feature_types.items() if ftype == data_type]
    
    def get_all_groups(self) -> List[str]:
        """
        Get all available feature groups.
        
        Returns:
            List of unique feature group names
        """
        return sorted(list(set(self.feature_groups.values())))
    
    def get_all_types(self) -> List[str]:
        """
        Get all available feature data types.
        
        Returns:
            List of unique feature data types
        """
        return sorted(list(set(self.feature_types.values())))
    
    def get_feature_count(self) -> int:
        """
        Get total number of features.
        
        Returns:
            Total feature count
        """
        return len(self.feature_mappings)
    
    def is_time_feature(self, feature_idx: int) -> bool:
        """
        Check if a feature is a time-related feature.
        
        Args:
            feature_idx: Feature index (0-127)
            
        Returns:
            True if time-related, False otherwise
        """
        feature_type = self.get_feature_type(feature_idx)
        return feature_type in ['time_ms', 'duration_ms']
    
    def is_coordinate_feature(self, feature_idx: int) -> bool:
        """
        Check if a feature is a coordinate feature.
        
        Args:
            feature_idx: Feature index (0-127)
            
        Returns:
            True if coordinate-related, False otherwise
        """
        feature_type = self.get_feature_type(feature_idx)
        return feature_type in ['world_coordinate', 'screen_coordinate']
    
    def get_feature_summary(self) -> Dict[str, Any]:
        """
        Get summary of all features.
        
        Returns:
            Dictionary with feature summary information
        """
        group_counts = {}
        type_counts = {}
        
        for group in self.feature_groups.values():
            group_counts[group] = group_counts.get(group, 0) + 1
        
        for ftype in self.feature_types.values():
            type_counts[ftype] = type_counts.get(ftype, 0) + 1
        
        return {
            'total_features': self.get_feature_count(),
            'groups': group_counts,
            'types': type_counts,
            'available_groups': self.get_all_groups(),
            'available_types': self.get_all_types()
        }
