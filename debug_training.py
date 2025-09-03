#!/usr/bin/env python3
"""
Debug script for training - allows you to test specific parts without full training
"""
import torch
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from ilbot.training.train_loop import run_training

def debug_training(sanity_mode=False):
    """Debug training with minimal configuration"""
    
    # Force CPU usage for debugging
    print(f"CUDA available: {torch.cuda.is_available()}")
    print("Forcing CPU usage for debugging")
    device = "cpu"
    
    if sanity_mode:
        print("H) SANITY MODE: Single forward+loss test")
    
    # Minimal config for debugging
    config = {
        "data_dir": "data/recording_sessions/20250831_113719/06_final_training_data",
        "targets_version": "v1",
        "enum_sizes": {},
        "epochs": 1,
        "lr": 0.00025,
        "weight_decay": 1e-4,
        "batch_size": 16,  # Small batch for debugging
        "disable_auto_batch": False,
        "grad_clip": None,
        "step_size": 8,
        "gamma": 0.5,
        "use_log1p_time": True,
        "time_div_ms": 1000.0,
        "time_clip_s": None,
        "loss_weights": {
            "time": 0.3, "x": 2.0, "y": 2.0,
            "type": 1.0, "btn": 1.0, "key": 1.0, "sx": 1.0, "sy": 1.0,
            "button": 1.0, "key_action": 1.0, "key_id": 1.0, "scroll_y": 1.0,
        },
        "seed": 1337,
        "device": device,
        "use_sequential": True,
    }
    
    print("Debug Training Configuration:")
    print(f"  Device: {device}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Epochs: {config['epochs']}")
    print(f"  Data dir: {config['data_dir']}")
    print("=" * 50)
    
    try:
        run_training(config)
        print("Debug training completed successfully!")
    except Exception as e:
        print(f"Debug training failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_training()
