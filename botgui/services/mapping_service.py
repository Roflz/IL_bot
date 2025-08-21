#!/usr/bin/env python3
"""Mapping service for hash/ID translations"""

import json
from pathlib import Path
from typing import Dict, Optional, Any
import logging

LOG = logging.getLogger(__name__)


class MappingService:
    """Service for translating feature values to human-readable labels"""
    
    def __init__(self, data_root: Path = Path("data")):
        self.data_root = data_root
        self.id_mappings: Dict[str, Any] = {}
        self._reverse_lookups: Dict[str, Dict[str, str]] = {}
        self._load_mappings()
        self._create_reverse_lookups()
    
    def _load_mappings(self):
        """Load ID mappings from both training data and live data, merging them"""
        # Load training mappings (base)
        training_mappings_file = self.data_root / "05_mappings" / "id_mappings.json"
        if training_mappings_file.exists():
            try:
                with open(training_mappings_file, 'r') as f:
                    self.id_mappings = json.load(f)
                LOG.info(f"Loaded training ID mappings with {len(self.id_mappings)} groups")
            except Exception as e:
                LOG.exception("Failed to load training ID mappings")
                self.id_mappings = {}
        else:
            LOG.warning(f"Training ID mappings file not found: {training_mappings_file}")
            self.id_mappings = {}
        
        # Load live mappings (overlay)
        live_mappings_file = self.data_root / "05_mappings" / "live_id_mappings.json"
        if live_mappings_file.exists():
            try:
                with open(live_mappings_file, 'r') as f:
                    live_mappings = json.load(f)
                
                # Merge live mappings with training mappings
                self._merge_mappings(live_mappings)
                LOG.info(f"Loaded and merged live ID mappings with {len(live_mappings)} groups")
            except Exception as e:
                LOG.exception("Failed to load live ID mappings")
        else:
            LOG.info("No live ID mappings file found, using training mappings only")
    
    def _merge_mappings(self, live_mappings: dict):
        # Add-only merge: keep training/default values; live only fills gaps
        for group, group_data in (live_mappings or {}).items():
            if not isinstance(group_data, dict):
                continue
            dst_group = self.id_mappings.setdefault(group, {})
            for mtype, live_table in group_data.items():
                if not isinstance(live_table, dict):
                    continue
                dst_table = dst_group.setdefault(mtype, {})
                for k, v in live_table.items():
                    k = str(k)
                    if k not in dst_table:
                        dst_table[k] = v
    
    def _create_reverse_lookups(self):
        """Create reverse lookup maps for fast translation"""
        self._reverse_lookups = {}
        LOG.info(f"Creating reverse lookups from {len(self.id_mappings)} mapping groups")
        
        # Process global mappings - these are the most important for live translations
        if "Global" in self.id_mappings:
            global_maps = self.id_mappings["Global"]
            for key, value in global_maps.items():
                if key.endswith("_hashes") or key.endswith("_ids"):
                    self._reverse_lookups[key] = self._build_reverse_map(value)
                elif key == "hash_mappings":
                    # Special case: hash_mappings contains direct hash->label mappings
                    self._reverse_lookups["Global.hash_mappings"] = self._build_reverse_map(value)
        
        # Process group mappings
        for group_name, group_data in self.id_mappings.items():
            if group_name == "Global":
                continue
            
            if isinstance(group_data, dict):
                for key, value in group_data.items():
                    if key.endswith("_hashes") or key.endswith("_ids"):
                        lookup_key = f"{group_name}.{key}"
                        self._reverse_lookups[lookup_key] = self._build_reverse_map(value)
        
        LOG.info(f"Created {len(self._reverse_lookups)} reverse lookup maps")
    
    def reload(self):
        """Reload mappings from disk and rebuild reverse lookups."""
        self._load_mappings()
        self._create_reverse_lookups()
    
    def _build_reverse_map(self, mapping_dict: Dict) -> Dict[str, str]:
        """Build reverse lookup map from forward mapping"""
        reverse_map = {}
        
        for key, value in mapping_dict.items():
            # The key is the ID/hash, the value is the human-readable label
            # We want to map ID/hash -> label
            if isinstance(key, (int, float)):
                # Convert numeric keys to strings for consistent lookup
                key_str = str(int(key)) if key == int(key) else str(key)
                reverse_map[key_str] = value
            else:
                reverse_map[str(key)] = value
        
        return reverse_map
    
    def translate(self, feature_idx: int, raw_value: Any,
                  group_hint: Optional[str] = None,
                  mapping_type_hint: Optional[str] = None) -> Optional[str]:
        if raw_value is None:
            return None
        try:
            key = str(int(raw_value)) if isinstance(raw_value, (int, float)) else str(raw_value)
        except Exception:
            key = str(raw_value)

        # Build search order
        order = []
        if group_hint and mapping_type_hint:
            order.append(f"{group_hint}.{mapping_type_hint}")
        if group_hint:
            order += [k for k in self._reverse_lookups.keys() if k.startswith(f"{group_hint}.")]
        if "Global.hash_mappings" in self._reverse_lookups:
            order.append("Global.hash_mappings")
        order += [k for k in self._reverse_lookups.keys() if k not in order]

        for name in order:
            rev = self._reverse_lookups.get(name, {})
            if key in rev:
                return rev[key]
        return None
    
    def get_feature_group_mappings(self, group_name: str) -> Dict[str, str]:
        """Get all mappings for a specific feature group"""
        if group_name == "All":
            return {}
        
        group_data = self.id_mappings.get(group_name, {})
        mappings = {}
        
        for key, value in group_data.items():
            if key.endswith("_hashes") or key.endswith("_ids"):
                if isinstance(value, dict):
                    for label, id_value in value.items():
                        key_str = str(int(id_value)) if isinstance(id_value, (int, float)) and id_value == int(id_value) else str(id_value)
                        mappings[key_str] = label
        
        return mappings
    
    def get_available_groups(self) -> list:
        """Get list of available feature groups"""
        groups = ["All"]
        for group_name in self.id_mappings.keys():
            if group_name != "Global":
                groups.append(group_name)
        return groups
