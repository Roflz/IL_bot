#!/usr/bin/env python3
"""
Debug script to test the exact same mapping loading and translation logic
that the GUI should be using.
"""

import json
import os

def test_gui_mapping_logic():
    """Test the exact same logic the GUI should be using"""
    
    print("=== DEBUGGING GUI MAPPING LOGIC ===\n")
    
    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print(f"Working directory: {os.getcwd()}")
    
    # 1. Test loading the mappings exactly like the GUI does
    print("1. LOADING MAPPINGS (exactly like GUI):")
    print("-" * 60)
    
    try:
        # Load feature mappings
        with open('data/features/feature_mappings.json', 'r') as f:
            live_feature_mappings = json.load(f)
        print(f"✅ live_feature_mappings: {len(live_feature_mappings)} features loaded")
        
        # Load ID mappings
        with open('data/features/id_mappings.json', 'r') as f:
            live_id_mappings = json.load(f)
        print(f"✅ live_id_mappings: {len(live_id_mappings)} groups loaded")
        
        # Create hash reverse lookup (exactly like GUI)
        hash_reverse_lookup = {}
        if live_id_mappings:
            # Global hash mappings
            if 'Global' in live_id_mappings and 'hash_mappings' in live_id_mappings['Global']:
                hash_mappings = live_id_mappings['Global']['hash_mappings']
                for hash_key, original_string in hash_mappings.items():
                    try:
                        hash_key_int = int(hash_key)
                        if 'hash_mappings' not in hash_reverse_lookup:
                            hash_reverse_lookup['hash_mappings'] = {}
                        hash_reverse_lookup['hash_mappings'][hash_key_int] = original_string
                    except (ValueError, TypeError):
                        pass
            
            # Feature-group-specific mappings
            for feature_group, group_mappings in live_id_mappings.items():
                if feature_group == 'Global':
                    continue
                
                for mapping_type, mappings in group_mappings.items():
                    if isinstance(mappings, dict):
                        for id_key, original_string in mappings.items():
                            try:
                                id_key_int = int(id_key)
                                if mapping_type not in hash_reverse_lookup:
                                    hash_reverse_lookup[mapping_type] = {}
                                hash_reverse_lookup[mapping_type][id_key_int] = original_string
                            except (ValueError, TypeError):
                                pass
        
        print(f"✅ hash_reverse_lookup: {len(hash_reverse_lookup)} mapping types created")
        
    except Exception as e:
        print(f"❌ Error loading mappings: {e}")
        return
    
    print("\n" + "=" * 60)
    
    # 2. Test the exact same format_value_for_display logic
    print("\n2. TESTING format_value_for_display LOGIC:")
    print("-" * 60)
    
    def format_value_for_display(value, feature_idx=None, show_translation=True):
        """Exact copy of the GUI's format_value_for_display method"""
        if isinstance(value, (int, float)):
            # Convert to float to handle numpy types
            value = float(value)
            
            # If showing translations, try to find a translation using the new ID mappings structure
            if show_translation and live_id_mappings:
                # Get feature info to determine the correct mapping category
                feature_name = None
                feature_group = None
                data_type = None
                
                if feature_idx is not None:
                    for feature_data in live_feature_mappings:
                        if isinstance(feature_data, dict) and feature_data.get('feature_index') == feature_idx:
                            feature_name = feature_data.get('feature_name')
                            feature_group = feature_data.get('feature_group')
                            data_type = feature_data.get('data_type')
                            break
                
                print(f"    🔍 Feature {feature_idx}: name='{feature_name}', group='{feature_group}', type='{data_type}'")
                
                # Handle boolean values automatically
                if data_type == 'boolean':
                    if value == 1.0:
                        return "true"
                    elif value == 0.0:
                        return "false"
                
                # Check feature-group-specific mappings
                if feature_group and feature_group in live_id_mappings:
                    group_mappings = live_id_mappings[feature_group]
                    print(f"    🔍 Found group '{feature_group}' with mappings: {list(group_mappings.keys())}")
                    
                    # Check hash mappings first (for hashed strings)
                    if 'hash_mappings' in group_mappings:
                        try:
                            hash_key = int(float(value))
                            if str(hash_key) in group_mappings['hash_mappings']:
                                original_value = group_mappings['hash_mappings'][str(hash_key)]
                                print(f"    ✅ Found in hash_mappings: {hash_key} -> '{original_value}'")
                                return str(original_value)
                        except (ValueError, TypeError):
                            pass
                    
                    # Check specific mapping types based on feature group
                    if feature_group == "Phase Context":
                        if 'phase_type_hashes' in group_mappings:
                            try:
                                hash_key = int(float(value))
                                if str(hash_key) in group_mappings['phase_type_hashes']:
                                    name = group_mappings['phase_type_hashes'][str(hash_key)]
                                    print(f"    ✅ Found in phase_type_hashes: {hash_key} -> '{name}'")
                                    return str(name)
                            except (ValueError, TypeError):
                                pass
                    
                    elif feature_group == "Inventory":
                        if 'item_ids' in group_mappings:
                            try:
                                id_key = int(float(value))
                                if str(id_key) in group_mappings['item_ids']:
                                    name = group_mappings['item_ids'][str(id_key)]
                                    print(f"    ✅ Found in item_ids: {id_key} -> '{name}'")
                                    return str(name)
                            except (ValueError, TypeError):
                                pass
                
                # Check global hash mappings as fallback
                if 'Global' in live_id_mappings and 'hash_mappings' in live_id_mappings['Global']:
                    try:
                        hash_key = int(float(value))
                        if str(hash_key) in live_id_mappings['Global']['hash_mappings']:
                            original_value = live_id_mappings['Global']['hash_mappings'][str(hash_key)]
                            print(f"    ✅ Found in Global hash_mappings: {hash_key} -> '{original_value}'")
                            return str(original_value)
                    except (ValueError, TypeError):
                        pass
                
                # Fallback: try the old feature-based lookup
                if show_translation and feature_idx is not None and feature_idx in hash_reverse_lookup:
                    if value in hash_reverse_lookup[feature_idx]:
                        original_value = hash_reverse_lookup[feature_idx][value]
                        print(f"    ✅ Found in hash_reverse_lookup[{feature_idx}]: {value} -> '{original_value}'")
                        return str(original_value)
                
                print(f"    ❌ No translation found for value {value}")
            
            # Handle special cases for non-translated values
            if value == 0:
                result = "0"
            elif value == -1:
                result = "-1"
            else:
                # If it's a whole number, show as integer (no .0)
                if value == int(value):
                    result = f"{int(value)}"
                else:
                    result = f"{value:.3f}"
            
            return result
        
        return str(value)
    
    # 3. Test specific translations
    print("\n3. TESTING SPECIFIC TRANSLATIONS:")
    print("-" * 60)
    
    # Test phase_type (feature index 63)
    print("\n🔍 Testing phase_type (feature index 63):")
    test_value = 40717.0
    result = format_value_for_display(test_value, feature_idx=63, show_translation=True)
    print(f"  Input: {test_value} -> Output: '{result}'")
    
    # Test inventory item (feature index 14)
    print("\n🔍 Testing inventory_slot_0 (feature index 14):")
    test_value = 1592.0
    result = format_value_for_display(test_value, feature_idx=14, show_translation=True)
    print(f"  Input: {test_value} -> Output: '{result}'")
    
    # Test with translations disabled
    print("\n🔍 Testing with translations disabled:")
    test_value = 40717.0
    result = format_value_for_display(test_value, feature_idx=63, show_translation=False)
    print(f"  Input: {test_value} -> Output: '{result}'")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    test_gui_mapping_logic()
