#!/usr/bin/env python3
"""
View example predictions from the trained OSRS Bot model
"""

import torch
import numpy as np
import argparse
from pathlib import Path
from setup_training import OSRSDataset, create_data_loaders
from model.imitation_hybrid_model import create_model

def load_trained_model(model_path: str, device: torch.device):
    """Load the trained model"""
    print(f"Loading trained model from: {model_path}")
    
    # Create model with same architecture
    model = create_model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model

def show_predictions(model, val_loader, device, num_examples=3):
    """Show example predictions vs actual targets"""
    print(f"\n{'='*80}")
    print("EXAMPLE PREDICTIONS")
    print(f"{'='*80}")
    
    model.eval()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if batch_idx >= num_examples:
                break
                
            # Get data
            temporal_sequence = batch['temporal_sequence'].to(device)
            action_sequence = batch['action_sequence'].to(device)
            action_target = batch['action_target'].to(device)
            
            # Get prediction
            prediction = model(temporal_sequence, action_sequence)
            
            print(f"\n--- Example {batch_idx + 1} ---")
            print(f"Input shapes:")
            print(f"  Gamestate: {temporal_sequence.shape}")
            print(f"  Actions: {action_sequence.shape}")
            print(f"  Target: {action_target.shape}")
            print(f"  Prediction: {prediction.shape}")
            
            # Show first few actions for comparison
            print(f"\nFirst 5 actions comparison:")
            print(f"{'Action':<6} {'Target':<25} {'Prediction':<25} {'Diff':<15}")
            print("-" * 75)
            
            for i in range(min(5, prediction.shape[1])):
                target = action_target[0, i].cpu().numpy()
                pred = prediction[0, i].cpu().numpy()
                
                # Format target and prediction
                target_str = f"[{target[0]:6.1f}, {target[1]:4.0f}, {target[2]:4.0f}, {target[3]:4.0f}]"
                pred_str = f"[{pred[0]:6.1f}, {pred[1]:4.0f}, {pred[2]:4.0f}, {pred[3]:4.0f}]"
                
                # Calculate difference
                diff = np.mean(np.abs(target - pred))
                
                print(f"{i:<6} {target_str:<25} {pred_str:<25} {diff:<15.3f}")
            
            # Show action type distribution
            print(f"\nAction type distribution:")
            target_types = action_target[0, :, 1].cpu().numpy()  # Action type is feature 1
            pred_types = prediction[0, :, 1].cpu().numpy()
            
            print(f"  Target types: {np.bincount(target_types.astype(int), minlength=4)}")
            print(f"  Pred types:   {np.bincount(pred_types.astype(int), minlength=4)}")
            
            # Show coordinate ranges
            print(f"\nCoordinate ranges:")
            target_x = action_target[0, :, 2].cpu().numpy()  # X coord is feature 2
            target_y = action_target[0, :, 3].cpu().numpy()  # Y coord is feature 3
            pred_x = prediction[0, :, 2].cpu().numpy()
            pred_y = prediction[0, :, 3].cpu().numpy()
            
            print(f"  Target X: [{target_x.min():.0f}, {target_x.max():.0f}]")
            print(f"  Target Y: [{target_y.min():.0f}, {target_y.max():.0f}]")
            print(f"  Pred X:   [{pred_x.min():.0f}, {pred_x.max():.0f}]")
            print(f"  Pred Y:   [{pred_y.min():.0f}, {pred_y.max():.0f}]")

def main():
    parser = argparse.ArgumentParser(description='View predictions from trained OSRS Bot model')
    parser.add_argument('--data_dir', type=str, default="data/recording_sessions/20250827_040359/06_final_training_data",
                       help='Path to training data directory')
    parser.add_argument('--model_path', type=str, default="training_results/model_weights.pth",
                       help='Path to trained model weights')
    parser.add_argument('--num_examples', type=int, default=3,
                       help='Number of examples to show')
    args = parser.parse_args()
    
    # Check files exist
    data_dir = Path(args.data_dir)
    model_path = Path(args.model_path)
    
    if not data_dir.exists():
        print(f"Error: Data directory {data_dir} does not exist!")
        return
        
    if not model_path.exists():
        print(f"Error: Model file {model_path} does not exist!")
        print("Please train the model first using train_model.py")
        return
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = load_trained_model(str(model_path), device)
    
    # Create data loaders
    print(f"\nLoading validation data from: {data_dir}")
    
    gamestate_file = data_dir / "gamestate_sequences.npy"
    action_input_file = data_dir / "action_input_sequences.npy"
    action_targets_file = data_dir / "action_targets.npy"
    
    dataset = OSRSDataset(
        gamestate_file=str(gamestate_file),
        action_input_file=str(action_input_file),
        action_targets_file=str(action_targets_file),
        sequence_length=10,
        max_actions=100
    )
    
    train_loader, val_loader = create_data_loaders(
        dataset=dataset,
        train_split=0.8,
        batch_size=1,  # Use batch size 1 for easier viewing
        shuffle=False,  # Don't shuffle for consistent examples
        device=device
    )
    
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Show predictions
    show_predictions(model, val_loader, device, args.num_examples)
    
    print(f"\n{'='*80}")
    print("PREDICTION VIEWING COMPLETE")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
