#!/usr/bin/env python3
"""
Simulate Live Execution from Recording CLI

This script simulates live execution by feeding recorded frames as if they were live,
demonstrating that the shared pipeline works for streaming scenarios.
"""

import sys
import argparse
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from shared_pipeline import (
    load_existing_features, load_action_targets, load_feature_mappings,
    normalize_features, create_temporal_sequences, validate_sequence_structure
)


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Simulate live execution using recorded data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simulate live execution with default settings
  python tools/simulate_live_from_recording.py
  
  # Custom data directory
  python tools/simulate_live_from_recording.py --data-dir /path/to/data
  
  # Simulate faster than real-time
  python tools/simulate_live_from_recording.py --speedup 2.0
        """
    )
    
    parser.add_argument(
        '--data-dir', 
        default='data',
        help='Root data directory (default: data)'
    )
    
    parser.add_argument(
        '--speedup',
        type=float,
        default=1.0,
        help='Speedup factor for simulation (default: 1.0 = real-time)'
    )
    
    parser.add_argument(
        '--max-frames',
        type=int,
        default=100,
        help='Maximum number of frames to simulate (default: 100)'
    )
    
    parser.add_argument(
        '--sequence-length',
        type=int,
        default=10,
        help='Sequence length for temporal context (default: 10)'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("SIMULATE LIVE EXECUTION FROM RECORDING")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")
    print(f"Speedup factor: {args.speedup}x")
    print(f"Max frames: {args.max_frames}")
    print(f"Sequence length: {args.sequence_length}")
    print("=" * 60)
    
    try:
        # Load existing data
        print("Loading recorded data...")
        features = load_existing_features(f"{args.data_dir}/features/state_features.npy")
        action_targets = load_action_targets(f"{args.data_dir}/training_data/action_targets.json")
        feature_mappings = load_feature_mappings(f"{args.data_dir}/features/feature_mappings.json")
        
        print(f"Loaded {len(features)} gamestates with {features.shape[1]} features each")
        print(f"Loaded {len(action_targets)} action target sequences")
        
        # Normalize features
        print("Normalizing features...")
        normalized_features = normalize_features(features, f"{args.data_dir}/features/feature_mappings.json")
        
        # Create temporal sequences
        print("Creating temporal sequences...")
        input_sequences, target_sequences, action_input_sequences = create_temporal_sequences(
            normalized_features, action_targets, args.sequence_length
        )
        
        # Validate sequence structure
        print("Validating sequence structure...")
        if not validate_sequence_structure(input_sequences, target_sequences, action_input_sequences):
            print("❌ Sequence validation failed!")
            return 1
        
        # Simulate live execution
        print("\n🚀 Starting live simulation...")
        simulate_live_execution(
            input_sequences, 
            target_sequences, 
            action_input_sequences,
            args.speedup,
            args.max_frames
        )
        
        print("\n✅ Live simulation completed successfully!")
        return 0
        
    except Exception as e:
        print(f"\n❌ Live simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


def simulate_live_execution(input_sequences, target_sequences, action_input_sequences, speedup, max_frames):
    """
    Simulate live execution by processing frames sequentially.
    
    Args:
        input_sequences: Input gamestate sequences
        target_sequences: Target action sequences
        action_input_sequences: Action input sequences for context
        speedup: Speedup factor for simulation
        max_frames: Maximum number of frames to process
    """
    n_sequences = min(len(input_sequences), max_frames)
    
    print(f"Simulating {n_sequences} frames at {speedup}x speed...")
    print("=" * 50)
    
    # Simulate processing each frame
    for i in range(n_sequences):
        start_time = time.time()
        
        # Simulate frame processing
        frame_data = {
            'frame_index': i,
            'gamestate_sequence': input_sequences[i],
            'action_sequence': action_input_sequences[i],
            'target_actions': target_sequences[i]
        }
        
        # Process frame (simulate model inference)
        process_frame(frame_data)
        
        # Calculate timing
        processing_time = time.time() - start_time
        
        # Simulate real-time constraints
        if speedup > 0:
            target_frame_time = 0.1 / speedup  # 100ms per frame at real-time
            sleep_time = max(0, target_frame_time - processing_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        # Progress indicator
        if (i + 1) % 10 == 0 or i == n_sequences - 1:
            progress = (i + 1) / n_sequences * 100
            print(f"Processed frame {i + 1}/{n_sequences} ({progress:.1f}%) - "
                  f"Processing: {processing_time*1000:.1f}ms")
    
    print("=" * 50)
    print(f"Simulation complete! Processed {n_sequences} frames")


def process_frame(frame_data):
    """
    Process a single frame (simulates model inference).
    
    Args:
        frame_data: Dictionary containing frame information
    """
    # Extract frame information
    frame_index = frame_data['frame_index']
    gamestate_sequence = frame_data['gamestate_sequence']
    action_sequence = frame_data['action_sequence']
    target_actions = frame_data['target_actions']
    
    # Simulate feature extraction (already done in this simulation)
    n_features = gamestate_sequence.shape[1]
    sequence_length = gamestate_sequence.shape[0]
    
    # Simulate action prediction (in real scenario, this would be model inference)
    predicted_actions = simulate_action_prediction(gamestate_sequence, action_sequence)
    
    # Simulate evaluation metrics
    accuracy = calculate_prediction_accuracy(predicted_actions, target_actions)
    
    # Log frame processing (verbose for first few frames)
    if frame_index < 3:
        print(f"\n📊 Frame {frame_index} Processing:")
        print(f"  - Gamestate sequence: {gamestate_sequence.shape}")
        print(f"  - Action sequence: {len(action_sequence)} timesteps")
        print(f"  - Target actions: {len(target_actions)} actions")
        print(f"  - Predicted actions: {len(predicted_actions)} actions")
        print(f"  - Prediction accuracy: {accuracy:.2f}")


def simulate_action_prediction(gamestate_sequence, action_sequence):
    """
    Simulate action prediction (placeholder for model inference).
    
    Args:
        gamestate_sequence: Input gamestate sequence
        action_sequence: Input action sequence
        
    Returns:
        Simulated predicted actions
    """
    # This is a placeholder - in real execution, this would be model inference
    # For simulation, we'll return a simple heuristic-based prediction
    
    # Extract current gamestate (last timestep)
    current_gamestate = gamestate_sequence[-1]
    
    # Simple heuristic: predict actions based on current state
    # This is just for demonstration - real model would be much more sophisticated
    
    # Simulate different action types based on gamestate features
    predicted_actions = []
    
    # Player position features (indices 0-1)
    player_x = current_gamestate[0]
    player_y = current_gamestate[1]
    
    # Animation feature (index 2)
    animation = current_gamestate[2]
    
    # Movement state (index 3)
    is_moving = current_gamestate[3]
    
    # Generate simulated predictions based on state
    if is_moving > 0.5:
        # Player is moving - predict movement actions
        predicted_actions = [
            [1, 0.0, 0, player_x + 10, player_y + 10, 0, 0, 0, 0]  # Move action
        ]
    elif animation > 0:
        # Player is performing action - predict interaction
        predicted_actions = [
            [1, 0.0, 1, player_x, player_y, 1, 0, 0, 0]  # Click action
        ]
    else:
        # Player is idle - predict no actions
        predicted_actions = [0]  # No actions
    
    return predicted_actions


def calculate_prediction_accuracy(predicted_actions, target_actions):
    """
    Calculate prediction accuracy (simplified).
    
    Args:
        predicted_actions: Predicted action sequence
        target_actions: Target action sequence
        
    Returns:
        Accuracy score between 0 and 1
    """
    if not target_actions or len(target_actions) < 1:
        return 0.0
    
    if not predicted_actions or len(predicted_actions) < 1:
        return 0.0
    
    # Simple accuracy: compare action counts
    target_count = int(target_actions[0]) if target_actions else 0
    predicted_count = int(predicted_actions[0]) if predicted_actions else 0
    
    if target_count == 0 and predicted_count == 0:
        return 1.0  # Perfect match for no actions
    
    if target_count == 0 or predicted_count == 0:
        return 0.0  # Mismatch if one has actions and other doesn't
    
    # For actions, check if we predicted the right number
    count_accuracy = 1.0 - abs(target_count - predicted_count) / max(target_count, 1)
    
    return max(0.0, count_accuracy)


if __name__ == "__main__":
    sys.exit(main())
