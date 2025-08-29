"""
Mapping Service

Provides hash/ID translation services using the id_mappings structure.
"""

from typing import Dict, Any, Optional, List


class MappingService:
    """Service for translating feature values to human-readable labels."""
    
    def __init__(self, id_mappings: Dict[str, Any]):
        """
        Initialize the mapping service.
        
        Args:
            id_mappings: Dictionary containing ID mappings from the pipeline
        """
        self.id_mappings = id_mappings
        self._create_reverse_lookups()
    
    def _create_reverse_lookups(self):
        """Create reverse lookup dictionaries for efficient translation."""
        self.hash_to_name = {}
        self.id_to_name = {}
        
        # Process global mappings (both 'global' and 'Global')
        for global_key in ['global', 'Global']:
            if global_key in self.id_mappings:
                global_maps = self.id_mappings[global_key]
                
                # Hash mappings
                if 'hashes' in global_maps:
                    for hash_val, name in global_maps['hashes'].items():
                        self.hash_to_name[hash_val] = name
                
                # ID mappings
                if 'ids' in global_maps:
                    for id_val, name in global_maps['ids'].items():
                        self.id_to_name[str(id_val)] = name
        
        # Also load from "hash_mappings" if it exists
        if 'hash_mappings' in self.id_mappings:
            hash_maps = self.id_mappings['hash_mappings']
            if isinstance(hash_maps, dict):
                self.hash_to_name.update(hash_maps)
        
        # Process group-specific mappings
        for group_name, group_data in self.id_mappings.items():
            if group_name in ['global', 'Global', 'hash_mappings']:
                continue
                
            if isinstance(group_data, dict):
                # Hash mappings
                if 'hashes' in group_data:
                    for hash_val, name in group_data['hashes'].items():
                        self.hash_to_name[hash_val] = name
                
                # ID mappings
                if 'ids' in group_data:
                    for id_val, name in group_data['ids'].items():
                        self.id_to_name[str(id_val)] = name
                
                # Absorb any dicts ending with "_hashes" or "_ids" into the reverse maps
                for key, value in group_data.items():
                    if isinstance(value, dict):
                        if key.endswith('_hashes'):
                            self.hash_to_name.update(value)
                        elif key.endswith('_ids'):
                            self.id_to_name.update(value)
    
    def translate(self, feature_idx: int, raw_value: Any) -> str:
        """
        Translate a raw feature value to a human-readable string.
        
        Args:
            feature_idx: Feature index (0-127)
            raw_value: Raw value from the feature array
            
        Returns:
            Translated string or original value as string
        """
        if raw_value is None:
            return "None"
        
        # Normalize keys: if raw_value is a float with no fractional part, use str(int(raw_value))
        # if int, use str(int)
        if isinstance(raw_value, float) and raw_value.is_integer():
            key_str = str(int(raw_value))
        elif isinstance(raw_value, int):
            key_str = str(int(raw_value))
        else:
            key_str = str(raw_value)
        
        # Search through all groups for the value
        for group_name, group_data in self.id_mappings.items():
            if isinstance(group_data, dict):
                # Check hashes
                if 'hashes' in group_data and key_str in group_data['hashes']:
                    result = group_data['hashes'][key_str]
                    return result
                
                # Check IDs
                if 'ids' in group_data and key_str in group_data['ids']:
                    result = group_data['ids'][key_str]
                    return result
                
                # Check other nested structures
                for key, value in group_data.items():
                    if isinstance(value, dict) and key_str in value:
                        result = value[key_str]
                        return result
        # Return original value as string
        return str(raw_value)
    
    def translate_bulk(self, feature_indices: List[int], raw_values: List[Any]) -> List[str]:
        """
        Translate multiple feature values at once.
        
        Args:
            feature_indices: List of feature indices
            raw_values: List of raw values
            
        Returns:
            List of translated strings
        """
        if len(feature_indices) != len(raw_values):
            raise ValueError("Feature indices and raw values must have same length")
        
        return [self.translate(idx, val) for idx, val in zip(feature_indices, raw_values)]
    
    def get_feature_names(self, feature_mappings: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        Extract feature names from feature mappings.
        
        Args:
            feature_mappings: List of feature mapping dictionaries
            
        Returns:
            Dictionary mapping feature index to feature name
        """
        feature_names = {}
        for mapping in feature_mappings:
            feature_idx = mapping.get('feature_index')
            feature_name = mapping.get('feature_name')
            if feature_idx is not None and feature_name is not None:
                feature_names[str(feature_idx)] = feature_name
        return feature_names
    
    def get_feature_groups(self, feature_mappings: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        Extract feature groups from feature mappings.
        
        Args:
            feature_mappings: List of feature mapping dictionaries
            
        Returns:
            Dictionary mapping feature index to feature group
        """
        feature_groups = {}
        for mapping in feature_mappings:
            feature_idx = mapping.get('feature_index')
            feature_group = mapping.get('feature_group')
            if feature_idx is not None and feature_group is not None:
                feature_groups[str(feature_idx)] = feature_group
        return feature_groups
