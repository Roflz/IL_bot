#!/usr/bin/env python3
"""
Action Distribution Analyzer
Analyzes the actual distribution of action types in the dataset.
"""

import numpy as np
import argparse
from pathlib import Path

def analyze_action_distribution(action_targets_path):
    """
    Analyze the distribution of action types in action_targets.npy
    
    Args:
        action_targets_path: Path to action_targets.npy file
    """
    print(f"🔍 Analyzing action distribution from: {action_targets_path}")
    print("=" * 60)
    
    # Load the data
    try:
        action_targets = np.load(action_targets_path)
        print(f"✅ Loaded action targets: {action_targets.shape}")
        print(f"  Shape: {action_targets.shape[0]} sequences × {action_targets.shape[1]} max actions × {action_targets.shape[2]} features")
    except Exception as e:
        print(f"❌ Failed to load {action_targets_path}: {e}")
        return
    
    # Extract action components
    # action_targets format: [B, A, 7] where 7 = [time, x, y, button, key_action, key_id, scroll_y]
    button_actions = action_targets[:, :, 3]  # Button column
    key_actions = action_targets[:, :, 4]     # Key action column  
    scroll_actions = action_targets[:, :, 6]  # Scroll column
    
    # IMPORTANT: We need to distinguish between actual actions and padding
    # Padding rows have ALL zeros in ALL columns
    # Real actions have at least one non-zero value in ANY column
    
    # Create mask for actual actions (not padding)
    has_action = np.any(action_targets != 0, axis=2)  # Any non-zero value in any column
    
    # Count actual actions vs padding
    total_possible = action_targets.shape[0] * action_targets.shape[1]
    actual_actions = np.count_nonzero(has_action)
    padding_rows = total_possible - actual_actions
    
    print(f"\n📊 Data Structure Analysis:")
    print(f"  Total possible positions: {total_possible:,}")
    print(f"  Actual actions: {actual_actions:,}")
    print(f"  Padding rows: {padding_rows:,}")
    print(f"  Padding percentage: {padding_rows/total_possible*100:.1f}%")
    
    # Now count only the actual actions
    actual_button = np.count_nonzero(button_actions[has_action])
    actual_key = np.count_nonzero(key_actions[has_action])
    actual_scroll = np.count_nonzero(scroll_actions[has_action])
    
    print(f"\n📊 Actual Action Counts (excluding padding):")
    print(f"  Button actions: {actual_button:,}")
    print(f"  Key actions: {actual_key:,}")
    print(f"  Scroll actions: {actual_scroll:,}")
    
    # A "move" action is when button=0, key_action=0, scroll=0, but time/coordinates exist
    # (not all zeros - that would be padding)
    move_mask = np.logical_and(button_actions[has_action] == 0, 
                              np.logical_and(key_actions[has_action] == 0, scroll_actions[has_action] == 0))
    move_events = np.count_nonzero(move_mask)
    
    print(f"\n🔍 Looking for move events (button=0, key=0, scroll=0, but has time/coordinates):")
    print(f"  Move events found: {move_events:,}")
    print(f"  Move events percentage: {move_events/actual_actions*100:.1f}%")
    
    if move_events > 0:
        print(f"  Sample move events:")
        move_indices = np.where(move_mask)[0]
        for i in range(min(3, len(move_indices))):
            idx = move_indices[i]
            action = action_targets[has_action][idx]
            print(f"    Move {i+1}: [time={action[0]:.1f}, x={action[1]:.0f}, y={action[2]:.0f}, "
                  f"button={action[3]:.0f}, key_action={action[4]:.0f}, key_id={action[5]:.0f}, scroll_y={action[6]:.0f}]")
    
    # Summary
    print(f"\n📋 Summary:")
    print(f"  Your dataset has {actual_actions:,} actual actions (excluding padding)")
    print(f"  Plus {padding_rows:,} padding rows ({padding_rows/total_possible*100:.1f}% of total positions)")
    print(f"  Button: {actual_button:,}")
    print(f"  Key: {actual_key:,}")
    print(f"  Scroll: {actual_scroll:,}")
    print(f"  Move: {move_events:,}")

def main():
    parser = argparse.ArgumentParser(description="Analyze action distribution in dataset")
    parser.add_argument("action_targets_path", help="Path to action_targets.npy file")
    
    args = parser.parse_args()
    
    if not Path(args.action_targets_path).exists():
        print(f"❌ File not found: {args.action_targets_path}")
        return
    
    analyze_action_distribution(args.action_targets_path)

if __name__ == "__main__":
    main()
