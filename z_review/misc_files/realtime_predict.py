#!/usr/bin/env python3
"""
Real-time prediction script for OSRS Bot - single sequence, no batching
Shows predictions on real training and validation data
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
    
    model = create_model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully!")
    return model

def predict_single_sequence(model, gamestate_sequence, action_sequence, device):
    """
    Make a single prediction for real-time use
    
    Args:
        gamestate_sequence: (10, 128) - 10 timesteps of gamestate features
        action_sequence: (10, 100, 8) - 10 timesteps of action sequences
        device: torch device
    
    Returns:
        prediction: (100, 8) - 100 predicted actions
    """
    model.eval()
    
    with torch.no_grad():
        # Add batch dimension for model input
        gamestate_batch = gamestate_sequence.unsqueeze(0)  # (1, 10, 128)
        action_batch = action_sequence.unsqueeze(0)        # (1, 10, 100, 8)
        
        # Get prediction
        prediction = model(gamestate_batch, action_batch)  # (1, 100, 8)
        
        # Remove batch dimension for real-time output
        prediction = prediction.squeeze(0)  # (100, 8)
        
        return prediction

def show_prediction_comparison(sequence_idx, gamestate_sequence, action_sequence, prediction, target, dataset_name):
    """Display prediction results with comparison to target"""
    print(f"\n{'='*80}")
    print(f"{dataset_name.upper()} PREDICTION #{sequence_idx + 1}")
    print(f"{'='*80}")
    
    print(f"Input shapes:")
    print(f"  Gamestate: {gamestate_sequence.shape} (10 timesteps, 128 features)")
    print(f"  Actions: {action_sequence.shape} (10 timesteps, 100 actions, 8 features)")
    print(f"  Prediction: {prediction.shape} (100 actions, 8 features)")
    print(f"  Target: {target.shape} (100 actions, 8 features)")
    
    print(f"\nFirst 10 actions - Prediction vs Target:")
    print(f"{'Action':<6} {'Time':<8} {'Type':<6} {'X':<6} {'Y':<6} {'Button':<8} {'Key':<6} {'Scroll':<8}")
    print(f"{'':<6} {'Pred':<8} {'Pred':<6} {'Pred':<6} {'Pred':<6} {'Pred':<8} {'Pred':<6} {'Pred':<8}")
    print("-" * 70)
    
    for i in range(min(10, prediction.shape[0])):
        pred = prediction[i].cpu().numpy()
        targ = target[i].cpu().numpy()
        
        print(f"{i:<6} {pred[0]:<8.1f} {pred[1]:<6.0f} {pred[2]:<6.0f} {pred[3]:<6.0f} {pred[4]:<8.0f} {pred[5]:<6.0f} {pred[6]:<6.0f} {pred[7]:<6.0f}")
        print(f"{'':<6} {targ[0]:<8.1f} {targ[1]:<6.0f} {targ[2]:<6.0f} {targ[3]:<6.0f} {targ[4]:<8.0f} {targ[5]:<6.0f} {targ[6]:<6.0f} {targ[7]:<6.0f}")
        print()
    
    # Calculate prediction accuracy
    pred_types = prediction[:, 1].cpu().numpy()
    target_types = target[:, 1].cpu().numpy()
    
    print(f"Action type comparison:")
    print(f"  Target types: {np.bincount(target_types.astype(int), minlength=4)}")
    print(f"  Pred types:   {np.bincount(pred_types.astype(int), minlength=4)}")
    
    # Calculate coordinate accuracy
    pred_x = prediction[:, 2].cpu().numpy()
    pred_y = prediction[:, 3].cpu().numpy()
    target_x = target[:, 2].cpu().numpy()
    target_y = target[:, 3].cpu().numpy()
    
    print(f"\nCoordinate comparison:")
    print(f"  Target X: [{target_x.min():.0f}, {target_x.max():.0f}]")
    print(f"  Pred X:   [{pred_x.min():.0f}, {pred_x.max():.0f}]")
    print(f"  Target Y: [{target_y.min():.0f}, {target_y.max():.0f}]")
    print(f"  Pred Y:   [{pred_y.min():.0f}, {pred_y.max():.0f}]")
    
    # Calculate overall accuracy (mean absolute error)
    mae = np.mean(np.abs(prediction.cpu().numpy() - target.cpu().numpy()))
    print(f"\nMean Absolute Error: {mae:.3f}")

def main():
    parser = argparse.ArgumentParser(description='Real-time prediction for OSRS Bot')
    parser.add_argument('--data_dir', type=str, default="data/recording_sessions/20250827_040359/06_final_training_data",
                       help='Path to training data directory')
    parser.add_argument('--model_path', type=str, default="training_results/model_weights.pth",
                       help='Path to trained model weights')
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
    
    # Load real training data
    print(f"\nLoading real training data from: {data_dir}")
    
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
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        dataset=dataset,
        train_split=0.8,
        batch_size=1,  # Use batch size 1 for single predictions
        shuffle=False,  # Don't shuffle for consistent examples
        device=device
    )
    
    print(f"Dataset loaded: {len(dataset)} total sequences")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Test predictions on training data
    print(f"\n{'='*80}")
    print("TESTING PREDICTIONS ON TRAINING DATA")
    print(f"{'='*80}")
    
    for i in range(3):
        if i < len(train_loader):
            batch = next(iter(train_loader))
            gamestate_sequence = batch['temporal_sequence'].squeeze(0)  # Remove batch dim
            action_sequence = batch['action_sequence'].squeeze(0)       # Remove batch dim
            target = batch['action_target'].squeeze(0)                 # Remove batch dim
            
            prediction = predict_single_sequence(model, gamestate_sequence, action_sequence, device)
            show_prediction_comparison(i, gamestate_sequence, action_sequence, prediction, target, "Training")
    
    # Test predictions on validation data
    print(f"\n{'='*80}")
    print("TESTING PREDICTIONS ON VALIDATION DATA")
    print(f"{'='*80}")
    
    for i in range(3):
        if i < len(val_loader):
            batch = next(iter(val_loader))
            gamestate_sequence = batch['temporal_sequence'].squeeze(0)  # Remove batch dim
            action_sequence = batch['action_sequence'].squeeze(0)       # Remove batch dim
            target = batch['action_target'].squeeze(0)                 # Remove batch dim
            
            prediction = predict_single_sequence(model, gamestate_sequence, action_sequence, device)
            show_prediction_comparison(i, gamestate_sequence, action_sequence, prediction, target, "Validation")
    
    print(f"\n{'='*80}")
    print("PREDICTION TESTING COMPLETE")
    print(f"{'='*80}")
    print(f"Training data: How well your model predicts on data it has seen")
    print(f"Validation data: How well your model generalizes to unseen data")
    print(f"Lower Mean Absolute Error = Better predictions!")

if __name__ == "__main__":
    main()
