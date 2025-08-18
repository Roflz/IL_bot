#!/usr/bin/env python3
"""
Comprehensive feature summary for manual review
Shows every feature with examples, context, and statistics
"""

import numpy as np
import json

def get_feature_name_and_context(feature_idx):
    """Get human-readable name and context for each feature from the actual data."""
    # Load the real feature information from feature_index_reference.json
    try:
        with open('data/features/feature_index_reference.json', 'r') as f:
            feature_ref = json.load(f)
        
        if str(feature_idx) in feature_ref:
            feature_data = feature_ref[str(feature_idx)]
            return feature_data['feature_name'], feature_data['context']
        else:
            return f"unknown_feature_{feature_idx}", "Unknown feature"
    except Exception as e:
        # Fallback to basic names if file can't be loaded
        fallback_names = {
            0: "player_world_x", 1: "player_world_y", 2: "player_animation_id", 
            3: "player_is_moving", 4: "player_movement_direction", 5: "last_interaction_combined",
            6: "camera_x", 7: "camera_y", 8: "camera_z", 9: "camera_pitch", 10: "camera_yaw",
            72: "timestamp"
        }
        name = fallback_names.get(feature_idx, f"feature_{feature_idx}")
        return name, "Feature from gamestate data"

def get_hash_mapping(feature_idx, hash_value, mappings_data):
    """Get the human-readable value that a hash represents."""
    try:
        # Look through the mappings data to find this hash value
        found_mappings = []
        
        for gamestate_idx, gamestate_mappings in enumerate(mappings_data):
            # Check if this gamestate has mappings and if they contain the feature we want
            if gamestate_mappings and len(gamestate_mappings) > feature_idx:
                for feature_mapping in gamestate_mappings:
                    if feature_mapping.get('feature_index') == feature_idx:
                        processed_val = feature_mapping.get('processed_value', 0)
                        if abs(processed_val - hash_value) < 0.001:
                            # Found the hash! Return the original value
                            original_value = feature_mapping.get('original_value', 'Unknown')
                            return str(original_value)
                        found_mappings.append(processed_val)
        
        # If not found, show what values we did find for this feature
        if found_mappings:
            unique_found = list(set(found_mappings))
            unique_found.sort()
            return f"Hash {hash_value} (not found, feature has: {unique_found[:5]})"
        else:
            return f"Hash {hash_value} (no mappings found for feature {feature_idx})"
            
    except Exception as e:
        return f"Error: {str(e)}"

def analyze_feature_for_summary(feature_idx, features_array, mappings_data):
    """Analyze a single feature for summary display."""
    feature_values = features_array[:, feature_idx]
    
    # Basic stats
    unique_count = len(np.unique(feature_values))
    min_val = np.min(feature_values)
    max_val = np.max(feature_values)
    mean_val = np.mean(feature_values)
    
    # Get sample values with improved selection for better representation
    total_samples = len(feature_values)
    if total_samples >= 13:
        # Improved sample selection: ensure we get representative samples
        sample_indices = []
        
        # Always include first and last samples
        sample_indices.extend([0, total_samples-1])
        
        # Include middle sample
        sample_indices.append(total_samples//2)
        
        # For features with low variability, include some non-zero samples if they exist
        if unique_count <= 5 and min_val != max_val:
            # Find some non-zero samples to show variety
            non_zero_indices = np.where(feature_values != 0)[0]
            if len(non_zero_indices) > 0:
                # Add a few non-zero samples, spread out
                step = max(1, len(non_zero_indices) // 3)
                for i in range(0, min(3, len(non_zero_indices)), step):
                    sample_indices.append(non_zero_indices[i])
        
        # Fill remaining slots with evenly distributed samples
        remaining_slots = 13 - len(sample_indices)
        if remaining_slots > 0:
            step = total_samples // (remaining_slots + 1)
            for i in range(1, remaining_slots + 1):
                idx = i * step
                if idx not in sample_indices and idx < total_samples:
                    sample_indices.append(idx)
        
        # Sort indices and ensure we have exactly 13 unique samples
        sample_indices = sorted(list(set(sample_indices)))
        if len(sample_indices) > 13:
            sample_indices = sample_indices[:13]
        elif len(sample_indices) < 13:
            # Fill remaining slots with sequential samples
            for i in range(total_samples):
                if i not in sample_indices and len(sample_indices) < 13:
                    sample_indices.append(i)
        
        sample_values = [feature_values[i] for i in sample_indices]
    else:
        sample_values = feature_values.tolist()
    
    # Get top 5 most common values with counts and hash mappings
    unique_vals, counts = np.unique(feature_values, return_counts=True)
    # Sort by count (descending) and get top 5
    sorted_indices = np.argsort(counts)[::-1]
    top_5_common = []
    for i in range(min(5, len(sorted_indices))):
        idx = sorted_indices[i]
        value = unique_vals[idx]
        count = counts[idx]
        percentage = (count / len(feature_values)) * 100
        
        # Try to find hash mapping for this value
        hash_mapping = get_hash_mapping(feature_idx, value, mappings_data)
        if hash_mapping:
            top_5_common.append(f"{value} ({count} times, {percentage:.1f}%) -> {hash_mapping}")
        else:
            top_5_common.append(f"{value} ({count} times, {percentage:.1f}%)")
    
    # Check for potential issues
    issues = []
    if unique_count == 1:
        issues.append("CONSTANT VALUE")
    elif unique_count < 5:
        issues.append(f"Low variability ({unique_count} unique)")
    
    # Get feature name and context
    feature_name, feature_context = get_feature_name_and_context(feature_idx)
    
    return {
        'feature_idx': feature_idx,
        'feature_name': feature_name,
        'feature_context': feature_context,
        'unique_count': unique_count,
        'min_val': min_val,
        'max_val': max_val,
        'mean_val': mean_val,
        'variance': max_val - min_val,
        'issues': issues,
        'sample_values': sample_values,
        'top_5_common': top_5_common,
        'dtype': str(feature_values.dtype)
    }

def main():
    print("=== COMPREHENSIVE FEATURE SUMMARY FOR REVIEW ===")
    print("=" * 80)
    
    # Load data
    try:
        features = np.load('data/features/state_features.npy')
        print(f"Data: {features.shape[0]} samples × {features.shape[1]} features")
        
        with open('data/features/feature_mappings.json', 'r') as f:
            mappings = json.load(f)
        print(f"Mappings: {len(mappings)} entries")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    print(f"\nReviewing all {features.shape[1]} features...")
    print("=" * 80)
    
    # Analyze each feature
    all_analyses = []
    
    for feature_idx in range(features.shape[1]):
        analysis = analyze_feature_for_summary(feature_idx, features, mappings)
        all_analyses.append(analysis)
        
        # Print feature summary
        print(f"\n--- Feature {feature_idx}: {analysis['feature_name']} ---")
        print(f"Context: {analysis['feature_context']}")
        print(f"Data Type: {analysis['dtype']}")
        print(f"Unique Values: {analysis['unique_count']}")
        print(f"Range: {analysis['min_val']:.6g} to {analysis['max_val']:.6g}")
        print(f"Mean: {analysis['mean_val']:.6g}")
        print(f"Variance: {analysis['variance']:.6g}")
        
        if analysis['issues']:
            print(f"Notes: {'; '.join(analysis['issues'])}")
        
        print(f"Top 5 Most Common Values:")
        for i, common_val in enumerate(analysis['top_5_common']):
            print(f"  {i+1}. {common_val}")
        
        print(f"Sample Values: {analysis['sample_values']}")
        print("-" * 60)
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("=== SUMMARY STATISTICS ===")
    
    constant_features = [a for a in all_analyses if a['unique_count'] == 1]
    low_variability_features = [a for a in all_analyses if 1 < a['unique_count'] < 5]
    normal_features = [a for a in all_analyses if a['unique_count'] >= 5]
    
    print(f"Total Features: {len(all_analyses)}")
    print(f"Constant Features (1 unique value): {len(constant_features)}")
    print(f"Low Variability Features (2-4 unique values): {len(low_variability_features)}")
    print(f"Normal Features (5+ unique values): {len(normal_features)}")
    
    if constant_features:
        print(f"\nConstant Features:")
        for feature in constant_features:
            print(f"  Feature {feature['feature_idx']}: {feature['feature_name']} = {feature['min_val']}")
    
    if low_variability_features:
        print(f"\nLow Variability Features:")
        for feature in low_variability_features:
            print(f"  Feature {feature['feature_idx']}: {feature['feature_name']} ({feature['unique_count']} unique values)")
    
    print(f"\nFeatures with Most Variability:")
    sorted_by_variability = sorted(all_analyses, key=lambda x: x['unique_count'], reverse=True)
    for i, feature in enumerate(sorted_by_variability[:10]):
        print(f"  {i+1}. Feature {feature['feature_idx']}: {feature['feature_name']} ({feature['unique_count']} unique values)")
    
    print("\n=== REVIEW COMPLETE ===")
    print("Check each feature above to confirm the data looks correct.")
    print("Pay special attention to constant features and low variability features.")

if __name__ == "__main__":
    main()
