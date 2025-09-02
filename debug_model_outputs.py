#!/usr/bin/env python3
"""
Debug script to check what the enhanced model is actually outputting
"""

import torch
import numpy as np
from ilbot.model.imitation_hybrid_model import create_model

def debug_model_outputs():
    print("🔍 Debugging Enhanced Model Outputs")
    print("=" * 50)
    
    # Create a mock feature specification
    feature_spec = {
        "group_indices": {
            "continuous": list(range(0, 50)),
            "boolean": list(range(50, 80)),
            "counts": list(range(80, 100)),
            "angles": list(range(100, 110)),
            "time": list(range(110, 128)),
            "categorical": []
        },
        "total_cat_vocab": 0,
        "cat_offsets": [],
        "unknown_index_per_field": []
    }
    
    # Create model
    data_config = {
        'gamestate_dim': 128,
        'max_actions': 100,
        'action_features': 7,
        'temporal_window': 10,
        'enum_sizes': {'button': 3, 'key_action': 3, 'key_id': 6, 'scroll': 3},
        'event_types': 4
    }
    
    model = create_model(data_config, feature_spec=feature_spec)
    
    # Create test inputs
    batch_size = 2
    temporal_seq = torch.randn(batch_size, 10, 128)
    action_seq = torch.randn(batch_size, 10, 100, 7)
    
    # Create realistic action sequence with per-gamestate timing
    # Each gamestate starts at 0ms and goes up to max 600ms
    for t in range(10):  # For each timestep
        cumulative_time = 0.0
        for a in range(3):  # 3 actions per gamestate
            if a == 0:
                delta_time = 0.05  # First action: 0.05s from start
            else:
                delta_time = 0.02  # Subsequent actions: 0.02s apart
            
            cumulative_time += delta_time
            if cumulative_time <= 0.6:  # Within 600ms gamestate
                action_seq[0, t, a, 0] = delta_time  # Delta time
                action_seq[0, t, a, 1] = 0.5 + 0.1 * a  # X coordinate
                action_seq[0, t, a, 2] = 0.3 + 0.1 * a  # Y coordinate
    
    # Create valid mask: first 3 actions are valid, rest are padded
    valid_mask = torch.zeros(batch_size, 100, dtype=torch.bool)
    valid_mask[:, :3] = True  # First 3 actions are valid
    
    print(f"Input shapes:")
    print(f"  temporal_sequence: {temporal_seq.shape}")
    print(f"  action_sequence: {action_seq.shape}")
    print(f"  valid_mask: {valid_mask.shape}")
    print(f"  Valid actions per batch: {valid_mask.sum(dim=1).numpy()}")
    
    # Test the forward pass
    try:
        outputs = model(temporal_seq, action_seq, valid_mask)
        
        print(f"\n✅ Forward pass successful!")
        print(f"Output shapes:")
        for key, value in outputs.items():
            print(f"  {key}: {value.shape}")
        
        # Check if outputs are actually different across action slots
        print(f"\n🔍 Checking if outputs are actually different across action slots:")
        
        # Check event logits
        event_logits = outputs['event_logits']  # [B, max_actions, event_types]
        print(f"\nEvent logits - checking if different slots produce different predictions:")
        for i in range(3):
            logits = event_logits[0, i].detach().numpy()
            print(f"  Action {i}: {logits}")
        
        # Check if they're actually different
        slot0 = event_logits[0, 0].detach()
        slot1 = event_logits[0, 1].detach()
        slot2 = event_logits[0, 2].detach()
        
        diff_01 = torch.norm(slot0 - slot1).item()
        diff_02 = torch.norm(slot0 - slot2).item()
        diff_12 = torch.norm(slot1 - slot2).item()
        
        print(f"\nDifferences between slots:")
        print(f"  Slot 0 vs 1: {diff_01:.6f}")
        print(f"  Slot 0 vs 2: {diff_02:.6f}")
        print(f"  Slot 1 vs 2: {diff_12:.6f}")
        
        if diff_01 < 0.001 and diff_02 < 0.001 and diff_12 < 0.001:
            print(f"❌ PROBLEM: All slots are producing identical outputs!")
            print(f"   This means the enhanced features are NOT working!")
        else:
            print(f"✅ GOOD: Slots are producing different outputs")
            print(f"   Enhanced features are working!")
        
        # Check sequence length prediction
        seq_length = outputs['sequence_length']  # [B]
        print(f"\nSequence length predictions: {seq_length.detach().numpy()}")
        print(f"Actual valid actions: {valid_mask.sum(dim=1).numpy()}")
        
        # Check timing predictions
        time_q = outputs['time_q']  # [B, max_actions, 3]
        print(f"\nTiming predictions:")
        for i in range(3):
            timing = time_q[0, i].detach().numpy()
            print(f"  Action {i}: {timing}")
        
        # Check if timing predictions are different
        time_diff_01 = torch.norm(time_q[0, 0] - time_q[0, 1]).item()
        time_diff_02 = torch.norm(time_q[0, 0] - time_q[0, 2]).item()
        
        print(f"\nTiming differences between slots:")
        print(f"  Slot 0 vs 1: {time_diff_01:.6f}")
        print(f"  Slot 0 vs 2: {time_diff_02:.6f}")
        
        if time_diff_01 < 0.001 and time_diff_02 < 0.001:
            print(f"❌ PROBLEM: All timing predictions are identical!")
        else:
            print(f"✅ GOOD: Timing predictions are different across slots")
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_model_outputs()
