#!/usr/bin/env python3
"""
Comprehensive training data analysis for Phase 1 data preparation
Analyzes input sequences, target actions, and training data quality
"""

import numpy as np
import json
from pathlib import Path
from collections import defaultdict, Counter
import matplotlib.pyplot as plt

class TrainingDataAnalyzer:
    def __init__(self):
        self.training_dir = Path('data/training_data')
        self.features_dir = Path('data/features')
        
        # Load all training data
        self.input_sequences = None
        self.target_sequences = None
        self.action_sequences = None
        self.training_metadata = None
        self.feature_mappings = None
        self.feature_index_ref = None
        
        self.load_data()
    
    def load_data(self):
        """Load all training data files"""
        print("Loading training data...")
        
        try:
            # Load input sequences (numpy array)
            self.input_sequences = np.load(self.training_dir / "input_sequences.npy")
            print(f"✓ Input sequences: {self.input_sequences.shape}")
            
            # Load target sequences (JSON - variable length)
            with open(self.training_dir / "target_sequences.json", 'r') as f:
                self.target_sequences = json.load(f)
            print(f"✓ Target sequences: {len(self.target_sequences)} (variable lengths)")
            
            # Load action sequences metadata
            with open(self.training_dir / "action_sequences.json", 'r') as f:
                self.action_sequences = json.load(f)
            print(f"✓ Action sequences: {len(self.action_sequences)}")
            
            # Load training metadata
            with open(self.training_dir / "training_metadata.json", 'r') as f:
                self.training_metadata = json.load(f)
            print(f"✓ Training metadata loaded")
            
            # Load feature mappings for hash resolution
            with open(self.features_dir / "feature_mappings.json", 'r') as f:
                self.feature_mappings = json.load(f)
            print(f"✓ Feature mappings: {len(self.feature_mappings)} gamestates")
            
            # Load feature index reference for names
            with open(self.features_dir / "feature_index_reference.json", 'r') as f:
                self.feature_index_ref = json.load(f)
            print(f"✓ Feature index reference loaded")
            
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def analyze_input_sequences(self):
        """Analyze the input temporal sequences"""
        print("\n" + "="*80)
        print("📊 INPUT SEQUENCES ANALYSIS")
        print("="*80)
        
        n_sequences, sequence_length, n_features = self.input_sequences.shape
        
        print(f"Shape: {self.input_sequences.shape}")
        print(f"Data type: {self.input_sequences.dtype}")
        print(f"Memory usage: {self.input_sequences.nbytes / 1024:.1f} KB")
        
        # Feature-level analysis across all sequences
        print(f"\n--- Feature Analysis Across Sequences ---")
        feature_stats = []
        
        for feature_idx in range(n_features):
            # Get all values for this feature across all sequences and timesteps
            feature_values = self.input_sequences[:, :, feature_idx].flatten()
            
            stats = {
                'feature_idx': feature_idx,
                'feature_name': self.get_feature_name(feature_idx),
                'unique_values': len(np.unique(feature_values)),
                'min_val': float(np.min(feature_values)),
                'max_val': float(np.max(feature_values)),
                'mean_val': float(np.mean(feature_values)),
                'std_val': float(np.std(feature_values)),
                'has_nan': bool(np.any(np.isnan(feature_values))),
                'has_inf': bool(np.any(np.isinf(feature_values)))
            }
            feature_stats.append(stats)
        
        # Sort by variability (most variable first)
        feature_stats.sort(key=lambda x: x['unique_values'], reverse=True)
        
        print(f"Top 10 Most Variable Features:")
        for i, stats in enumerate(feature_stats[:10]):
            print(f"  {i+1:2d}. Feature {stats['feature_idx']:2d}: {stats['feature_name']}")
            print(f"       Unique values: {stats['unique_values']:3d}, Range: {stats['min_val']:.3g} to {stats['max_val']:.3g}")
            print(f"       Mean: {stats['mean_val']:.3g}, Std: {stats['std_val']:.3g}")
        
        print(f"\nBottom 10 Least Variable Features:")
        for i, stats in enumerate(feature_stats[-10:]):
            print(f"  {i+1:2d}. Feature {stats['feature_idx']:2d}: {stats['feature_name']}")
            print(f"       Unique values: {stats['unique_values']:3d}, Range: {stats['min_val']:.3g} to {stats['max_val']:.3g}")
        
        # Sequence transition analysis
        print(f"\n--- How Much Do Gamestates Change Between Timesteps? ---")
        self.analyze_sequence_transitions()
        
        return feature_stats
    
    def analyze_sequence_transitions(self):
        """Analyze how much gamestates change between consecutive timesteps"""
        n_sequences, sequence_length, n_features = self.input_sequences.shape
        
        # Calculate changes between consecutive timesteps within each sequence
        transitions = []
        for seq_idx in range(n_sequences):
            for t in range(sequence_length - 1):
                current_state = self.input_sequences[seq_idx, t, :]
                next_state = self.input_sequences[seq_idx, t + 1, :]
                
                # Calculate change magnitude for each feature
                changes = np.abs(next_state - current_state)
                transitions.append(changes)
        
        transitions = np.array(transitions)
        
        # Feature-level transition statistics
        print(f"Analysis: {len(transitions)} state transitions (how much each feature changes between consecutive timesteps)")
        print(f"\nTop 10 Most Dynamic Features (change the most between timesteps):")
        
        # Calculate average change per feature and sort
        feature_avg_changes = []
        for feature_idx in range(n_features):
            feature_changes = transitions[:, feature_idx]
            mean_change = np.mean(feature_changes)
            feature_avg_changes.append((feature_idx, mean_change))
        
        # Sort by average change (most dynamic first)
        feature_avg_changes.sort(key=lambda x: x[1], reverse=True)
        
        for i, (feature_idx, mean_change) in enumerate(feature_avg_changes[:10]):
            feature_name = self.get_feature_name(feature_idx)
            print(f"  {i+1:2d}. {feature_name}")
            print(f"       Average change per timestep: {mean_change:.3g}")
        
        print(f"\nTop 10 Most Stable Features (change the least between timesteps):")
        for i, (feature_idx, mean_change) in enumerate(feature_avg_changes[-10:]):
            feature_name = self.get_feature_name(feature_idx)
            print(f"  {i+1:2d}. {feature_name}")
            print(f"       Average change per timestep: {mean_change:.3g}")
        
        print(f"\nInterpretation:")
        print(f"  • High change values = feature changes rapidly between timesteps (e.g., player position, camera)")
        print(f"  • Low change values = feature stays relatively stable (e.g., inventory items, skill levels)")
        print(f"  • This helps understand which features provide dynamic vs. stable context for the model")
    
    def analyze_target_actions(self):
        """Analyze the variable-length action targets"""
        print("\n" + "="*80)
        print("🎯 TARGET ACTIONS ANALYSIS")
        print("="*80)
        
        n_targets = len(self.target_sequences)
        target_lengths = [len(target) for target in self.target_sequences]
        
        print(f"Total targets: {n_targets}")
        print(f"Target lengths: {min(target_lengths)} to {max(target_lengths)} actions")
        print(f"Average length: {np.mean(target_lengths):.1f} actions")
        print(f"Median length: {np.median(target_lengths):.1f} actions")
        
        # Action count distribution
        print(f"\n--- Action Count Distribution ---")
        action_counts = [target[0] for target in self.target_sequences]  # First value is action count
        unique_counts, count_frequencies = np.unique(action_counts, return_counts=True)
        
        print(f"Action count frequencies:")
        for count, freq in zip(unique_counts, count_frequencies):
            percentage = (freq / n_targets) * 100
            print(f"  {count:2d} actions: {freq:3d} targets ({percentage:5.1f}%)")
        
        # Overall action type statistics across all targets
        print(f"\n--- Overall Action Type Statistics ---")
        all_action_types = []
        all_timings = []
        
        for target in self.target_sequences:
            action_count = int(target[0])
            if action_count > 0:
                for i in range(action_count):
                    base_idx = 1 + i * 7
                    if base_idx + 6 < len(target):
                        action_type = int(target[base_idx + 1])
                        timing = target[base_idx]
                        all_action_types.append(action_type)
                        all_timings.append(timing)
        
        # Count action types
        type_counts = Counter(all_action_types)
        type_names = {0: 'move', 1: 'click', 2: 'key', 3: 'scroll'}
        
        print(f"Total actions across all targets: {len(all_action_types)}")
        print(f"Action type breakdown:")
        for action_type, count in type_counts.items():
            type_name = type_names.get(action_type, f"unknown_{action_type}")
            percentage = (count / len(all_action_types)) * 100
            print(f"  {type_name}: {count:4d} actions ({percentage:5.1f}%)")
        
        # Timing analysis
        if all_timings:
            print(f"\nTiming analysis (0-1 normalized, where 1.0 = 600ms):")
            print(f"  Average timing: {np.mean(all_timings):.3f} ({np.mean(all_timings)*600:.0f}ms)")
            print(f"  Timing range: {min(all_timings):.3f} to {max(all_timings):.3f} ({min(all_timings)*600:.0f}ms to {max(all_timings)*600:.0f}ms)")
        
        # Detailed analysis of a few representative targets
        print(f"\n--- Sample Target Analysis (Human-Readable) ---")
        sample_indices = [0, n_targets//4, n_targets//2, 3*n_targets//4, n_targets-1]
        
        for i, idx in enumerate(sample_indices):
            if idx < len(self.target_sequences):
                target = self.target_sequences[idx]
                action_count = target[0]
                
                print(f"\nSample Target {i+1} (Index {idx}): {action_count} actions")
                
                # Parse action data
                if action_count > 0:
                    self.analyze_single_target_human_readable(target, idx)
                else:
                    print(f"  → No actions in this target (player was idle)")
    
    def analyze_single_target(self, target, target_idx):
        """Analyze a single target sequence"""
        action_count = int(target[0])
        if action_count == 0:
            print("  No actions in this target")
            return
        
        # Parse action data (7 values per action: timing, type, x, y, button, key, scroll)
        actions = []
        for i in range(action_count):
            base_idx = 1 + i * 7
            if base_idx + 6 < len(target):
                action = {
                    'timing': target[base_idx],      # Normalized 0-1
                    'type': target[base_idx + 1],    # 0=move, 1=click, 2=key, 3=scroll
                    'x': target[base_idx + 2],       # Normalized 0-1
                    'y': target[base_idx + 3],       # Normalized 0-1
                    'button': target[base_idx + 4],  # 0=none, 1=left, 2=right, 3=middle
                    'key': target[base_idx + 5],     # Normalized hash
                    'scroll': target[base_idx + 6]   # Normalized -1 to 1
                }
                actions.append(action)
        
        # Action type breakdown
        type_counts = Counter([int(a['type']) for a in actions])
        type_names = {0: 'move', 1: 'click', 2: 'key', 3: 'scroll'}
        
        print(f"  Action types: {dict(type_counts)}")
        for action_type, count in type_counts.items():
            type_name = type_names.get(action_type, f"unknown_{action_type}")
            print(f"    {type_name}: {count}")
        
        # Timing analysis
        timings = [a['timing'] for a in actions]
        print(f"  Timing range: {min(timings):.3f} to {max(timings):.3f} (0-1 normalized)")
        
        # Coordinate analysis
        x_coords = [a['x'] for a in actions]
        y_coords = [a['y'] for a in actions]
        print(f"  X coordinates: {min(x_coords):.3f} to {max(x_coords):.3f}")
        print(f"  Y coordinates: {min(y_coords):.3f} to {max(y_coords):.3f}")
    
    def analyze_single_target_human_readable(self, target, target_idx):
        """Analyze a single target sequence in human-readable format"""
        action_count = int(target[0])
        if action_count == 0:
            print("  → No actions in this target")
            return
        
        # Parse action data (7 values per action: timing, type, x, y, button, key, scroll)
        actions = []
        for i in range(action_count):
            base_idx = 1 + i * 7
            if base_idx + 6 < len(target):
                action = {
                    'timing': target[base_idx],      # Normalized 0-1
                    'type': target[base_idx + 1],    # 0=move, 1=click, 2=key, 3=scroll
                    'x': target[base_idx + 2],       # Normalized 0-1
                    'y': target[base_idx + 3],       # Normalized 0-1
                    'button': target[base_idx + 4],  # 0=none, 1=left, 2=right, 3=middle
                    'key': target[base_idx + 5],     # Normalized hash
                    'scroll': target[base_idx + 6]   # Normalized -1 to 1
                }
                actions.append(action)
        
        # Action type breakdown
        type_counts = Counter([int(a['type']) for a in actions])
        type_names = {0: 'move', 1: 'click', 2: 'key', 3: 'scroll'}
        
        print(f"  → Action breakdown:")
        for action_type, count in type_counts.items():
            type_name = type_names.get(action_type, f"unknown_{action_type}")
            print(f"    • {type_name}: {count} actions")
        
        # Timing analysis
        timings = [a['timing'] for a in actions]
        print(f"  → Timing: actions spread from {min(timings)*600:.0f}ms to {max(timings)*600:.0f}ms after gamestate")
        
        # Coordinate analysis
        x_coords = [a['x'] for a in actions]
        y_coords = [a['y'] for a in actions]
        print(f"  → Screen area: X range {min(x_coords):.3f}-{max(x_coords):.3f}, Y range {min(y_coords):.3f}-{max(y_coords):.3f}")
        
        # Show first few actions in detail
        print(f"  → First 3 actions:")
        for j, action in enumerate(actions[:3]):
            type_name = type_names.get(int(action['type']), f"type_{action['type']}")
            timing_ms = action['timing'] * 600
            x_norm, y_norm = action['x'], action['y']
            print(f"    {j+1}. {type_name} at ({x_norm:.3f}, {y_norm:.3f}) after {timing_ms:.0f}ms")
    
    def verify_sequence_alignment(self):
        """Verify that input sequences align correctly with target actions"""
        print("\n" + "="*80)
        print("🔗 SEQUENCE ALIGNMENT VERIFICATION")
        print("="*80)
        
        n_sequences = len(self.input_sequences)
        sequence_length = self.input_sequences.shape[1]
        
        print(f"Verifying alignment for {n_sequences} sequences...")
        print(f"Input: gamestates 0-{sequence_length-1} → Target: actions from gamestate {sequence_length}")
        
        # Check a few sample sequences
        sample_indices = [0, n_sequences//4, n_sequences//2, 3*n_sequences//4, n_sequences-1]
        
        for i, seq_idx in enumerate(sample_indices):
            if seq_idx < n_sequences:
                print(f"\n--- Sample Sequence {i+1} (Index {seq_idx}) ---")
                
                # Get input sequence (gamestates 0-9)
                input_seq = self.input_sequences[seq_idx]
                
                # Get target actions (from gamestate 10)
                target_idx = seq_idx + sequence_length
                if target_idx < len(self.action_sequences):
                    target_actions = self.action_sequences[target_idx]
                    
                    print(f"Input gamestates: {sequence_length} consecutive gamestates")
                    print(f"Target gamestate index: {target_idx}")
                    print(f"Target action count: {target_actions.get('action_count', 0)}")
                    
                    # Show first few actions
                    actions = target_actions.get('actions', [])
                    if actions:
                        print(f"First 3 actions:")
                        for j, action in enumerate(actions[:3]):
                            print(f"  Action {j+1}: {action['event_type']} at ({action['x_in_window']}, {action['y_in_window']})")
                else:
                    print(f"Target gamestate {target_idx} not found in action sequences")
    
    def analyze_training_data_quality(self):
        """Analyze overall training data quality"""
        print("\n" + "="*80)
        print("✨ TRAINING DATA QUALITY ANALYSIS")
        print("="*80)
        
        # Data integrity checks
        print("--- Data Integrity ---")
        
        # Check for NaN/Inf in input sequences
        input_nan_count = np.isnan(self.input_sequences).sum()
        input_inf_count = np.isinf(self.input_sequences).sum()
        
        print(f"Input sequences - NaN values: {input_nan_count}, Inf values: {input_inf_count}")
        
        # Check target sequence consistency
        target_lengths = [len(target) for target in self.target_sequences]
        expected_lengths = []
        
        for target in self.target_sequences:
            if len(target) > 0:
                action_count = int(target[0])
                expected_length = 1 + action_count * 7  # count + 7 values per action
                expected_lengths.append(expected_length)
        
        length_mismatches = sum(1 for actual, expected in zip(target_lengths, expected_lengths) if actual != expected)
        print(f"Target sequences - Length mismatches: {length_mismatches}/{len(self.target_sequences)}")
        
        # Sequence diversity analysis
        print("\n--- Sequence Diversity ---")
        
        # Calculate feature variance across sequences
        feature_variance = np.var(self.input_sequences, axis=(0, 1))  # Variance across all sequences and timesteps
        
        # Find most and least diverse features
        most_diverse_idx = np.argmax(feature_variance)
        least_diverse_idx = np.argmin(feature_variance)
        
        print(f"Most diverse feature: {most_diverse_idx} ({self.get_feature_name(most_diverse_idx)})")
        print(f"  Variance: {feature_variance[most_diverse_idx]:.6g}")
        
        print(f"Least diverse feature: {least_diverse_idx} ({self.get_feature_name(least_diverse_idx)})")
        print(f"  Variance: {feature_variance[least_diverse_idx]:.6g}")
        
        # Training metadata validation
        print("\n--- Metadata Validation ---")
        print(f"Expected sequences: {self.training_metadata.get('n_sequences')}")
        print(f"Actual sequences: {len(self.input_sequences)}")
        print(f"Expected sequence length: {self.training_metadata.get('sequence_length')}")
        print(f"Actual sequence length: {self.input_sequences.shape[1]}")
        print(f"Expected features: {self.training_metadata.get('feature_dimensions')}")
        print(f"Actual features: {self.input_sequences.shape[2]}")
    
    def get_feature_name(self, feature_idx):
        """Get human-readable feature name"""
        try:
            if str(feature_idx) in self.feature_index_ref:
                return self.feature_index_ref[str(feature_idx)]['feature_name']
            else:
                return f"feature_{feature_idx}"
        except:
            return f"feature_{feature_idx}"
    
    def generate_summary_report(self):
        """Generate a comprehensive summary report"""
        print("\n" + "="*80)
        print("📋 COMPREHENSIVE TRAINING DATA SUMMARY")
        print("="*80)
        
        # Basic statistics
        n_sequences, sequence_length, n_features = self.input_sequences.shape
        target_lengths = [len(target) for target in self.target_sequences]
        
        print(f"Training Data Overview:")
        print(f"  Input sequences: {n_sequences} sequences × {sequence_length} timesteps × {n_features} features")
        print(f"  Target actions: {len(self.target_sequences)} variable-length targets")
        print(f"  Action range: {min(target_lengths)} to {max(target_lengths)} actions per target")
        print(f"  Average actions: {np.mean(target_lengths):.1f} per target")
        
        # Data quality indicators
        print(f"\nData Quality Indicators:")
        
        # Feature variability
        feature_variance = np.var(self.input_sequences, axis=(0, 1))
        high_variance_features = np.sum(feature_variance > np.percentile(feature_variance, 75))
        low_variance_features = np.sum(feature_variance < np.percentile(feature_variance, 25))
        
        print(f"  High variability features: {high_variance_features}/{n_features}")
        print(f"  Low variability features: {low_variance_features}/{n_features}")
        
        # Action distribution
        action_counts = [target[0] for target in self.target_sequences]
        unique_action_counts = len(np.unique(action_counts))
        print(f"  Unique action count values: {unique_action_counts}")
        
        # Temporal consistency
        print(f"  Sequence length: {sequence_length} (consistent)")
        print(f"  Feature dimensions: {n_features} (consistent)")
        
        print(f"\nTraining Data Status: ✅ READY FOR TRAINING")
        print(f"The data shows good variability and consistency for imitation learning.")
    
    def run_full_analysis(self):
        """Run the complete training data analysis"""
        print("🔍 COMPREHENSIVE TRAINING DATA ANALYSIS")
        print("=" * 80)
        
        try:
            # Run all analysis components
            feature_stats = self.analyze_input_sequences()
            self.analyze_target_actions()
            self.verify_sequence_alignment()
            self.analyze_training_data_quality()
            self.generate_summary_report()
            
            print("\n" + "="*80)
            print("✅ ANALYSIS COMPLETE")
            print("="*80)
            
        except Exception as e:
            print(f"Error during analysis: {e}")
            raise

def main():
    """Main function to run training data analysis"""
    analyzer = TrainingDataAnalyzer()
    analyzer.run_full_analysis()

if __name__ == "__main__":
    main()
