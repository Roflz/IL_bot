#!/usr/bin/env python3
"""
Debug script to run the feature pipeline on a recording session to see what's happening with action extraction.
"""

import sys
import os
import logging
from pathlib import Path

# Add the current directory to Python path so we can import modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set up logging to see debug output
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

from botgui.services.feature_pipeline import FeaturePipeline

def debug_feature_pipeline():
    """Debug the feature pipeline on the current recording session."""
    
    # Path to the recording session
    session_dir = Path("data/recording_sessions/20250824_062052")
    
    # Check if the session exists
    if not session_dir.exists():
        print(f"Session directory does not exist: {session_dir}")
        return
    
    print(f"Debugging feature pipeline for session: {session_dir}")
    
    # Check if actions.csv exists
    actions_csv = session_dir / "actions.csv"
    if not actions_csv.exists():
        print(f"Actions CSV does not exist: {actions_csv}")
        return
    
    print(f"Actions CSV exists: {actions_csv}")
    print(f"Actions CSV size: {actions_csv.stat().st_size} bytes")
    
    # Check if gamestates directory exists
    gamestates_dir = session_dir / "gamestates"
    if not gamestates_dir.exists():
        print(f"Gamestates directory does not exist: {gamestates_dir}")
        return
    
    gamestate_files = list(gamestates_dir.glob("*.json"))
    print(f"Found {len(gamestate_files)} gamestate files")
    
    # Check if prepared directory exists
    prepared_dir = session_dir / "prepared"
    if prepared_dir.exists():
        print(f"Prepared directory already exists: {prepared_dir}")
        print("Removing it to force re-processing...")
        import shutil
        shutil.rmtree(prepared_dir)
    
    # Create feature pipeline
    print("Creating feature pipeline...")
    feature_pipeline = FeaturePipeline(Path("data"))
    
    # Prepare training data
    print("Preparing training data...")
    try:
        out_dir = feature_pipeline.prepare_session_training_data(session_dir)
        print(f"Training data prepared successfully at: {out_dir}")
        
        # Check the results
        if out_dir.exists():
            print(f"Output directory exists: {out_dir}")
            output_files = list(out_dir.glob("*"))
            print(f"Output files: {[f.name for f in output_files]}")
            
            # Check the action arrays
            y_actions_file = out_dir / "y_actions_norm_padded.npy"
            if y_actions_file.exists():
                import numpy as np
                y_actions = np.load(y_actions_file)
                print(f"y_actions shape: {y_actions.shape}")
                print(f"y_actions non-zero count: {np.count_nonzero(y_actions)}")
                print(f"y_actions max value: {y_actions.max()}")
                print(f"y_actions sample (first 5 rows):")
                print(y_actions[0, :5, :])
            
            action_inputs_file = out_dir / "action_input_sequences_padded.npy"
            if action_inputs_file.exists():
                import numpy as np
                action_inputs = np.load(action_inputs_file)
                print(f"action_inputs shape: {action_inputs.shape}")
                print(f"action_inputs non-zero count: {np.count_nonzero(action_inputs)}")
                print(f"action_inputs max value: {action_inputs.max()}")
        
    except Exception as e:
        print(f"Error preparing training data: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_feature_pipeline()
