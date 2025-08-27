#!/usr/bin/env python3
"""
Training script for OSRS Imitation Learning Model
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json
import time
import argparse
from pathlib import Path
from setup_training import create_data_loaders, setup_model, setup_training

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cpu'):
    """Train the model"""
    
    print(f"Starting training on {device}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"Initial CUDA memory: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB allocated, {torch.cuda.memory_reserved(0) / 1024**2:.1f} MB reserved")
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Epochs: {num_epochs}")
    print("=" * 60)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("Training...")
        
        for batch_idx, batch in enumerate(train_loader):
            # Move data to device
            temporal_sequence = batch['temporal_sequence'].to(device)
            action_sequence = batch['action_sequence'].to(device)
            action_target = batch['action_target'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            output = model(temporal_sequence, action_sequence)
            loss = criterion(output, action_target)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
            
            # Progress update
            if (batch_idx + 1) % 5 == 0:
                print(f"  Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        avg_train_loss = train_loss / train_batches
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        print("Validating...")
        
        with torch.no_grad():
            for batch in val_loader:
                temporal_sequence = batch['temporal_sequence'].to(device)
                action_sequence = batch['action_sequence'].to(device)
                action_target = batch['action_target'].to(device)
                
                output = model(temporal_sequence, action_sequence)
                loss = criterion(output, action_target)
                
                val_loss += loss.item()
                val_batches += 1
        
        avg_val_loss = val_loss / val_batches
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1} Summary:")
        print(f"  Training Loss: {avg_train_loss:.4f}")
        print(f"  Validation Loss: {avg_val_loss:.4f}")
        
        # CUDA memory management
        if torch.cuda.is_available():
            print(f"  CUDA Memory: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB allocated, {torch.cuda.memory_reserved(0) / 1024**2:.1f} MB reserved")
            # Clear cache to free up memory
            torch.cuda.empty_cache()
        
        print("-" * 40)
    
    return train_losses, val_losses

def save_training_results(model, train_losses, val_losses, config):
    """Save training results and model"""
    
    # Create results directory
    results_dir = Path("training_results")
    results_dir.mkdir(exist_ok=True)
    
    # Save training history
    training_history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'config': config,
        'final_train_loss': train_losses[-1] if train_losses else None,
        'final_val_loss': val_losses[-1] if val_losses else None,
        'best_val_loss': min(val_losses) if val_losses else None
    }
    
    with open(results_dir / "training_history.json", 'w') as f:
        json.dump(training_history, f, indent=2)
    
    # Save model
    torch.save(model.state_dict(), results_dir / "model_weights.pth")
    
    print(f"Training results saved to {results_dir}/")
    print(f"Model weights saved to {results_dir}/model_weights.pth")

def optimize_cuda_settings():
    """Optimize CUDA settings for better performance"""
    if torch.cuda.is_available():
        # Enable cuDNN benchmarking for faster convolutions
        torch.backends.cudnn.benchmark = True
        # Enable cuDNN deterministic mode for reproducibility
        torch.backends.cudnn.deterministic = False
        print("CUDA optimizations enabled")

def main():
    """Main training function"""
    
    print("OSRS Imitation Learning Model Training")
    print("=" * 60)
    
    # Optimize CUDA settings
    optimize_cuda_settings()
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"CUDA compute capability: {torch.cuda.get_device_capability(0)}")
    else:
        print("CUDA not available, using CPU")
    print(f"Device: {device}")
    
    # Load training setup
    print("Loading training setup...")
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train OSRS Bot Imitation Learning Model')
    parser.add_argument('--data_dir', type=str, default="data/06_final_training_data",
                       help='Path to training data directory (default: data/06_final_training_data)')
    args = parser.parse_args()
    
    # Create dataset and data loaders
    data_dir = Path(args.data_dir)
    gamestate_file = data_dir / "gamestate_sequences.npy"
    action_input_file = data_dir / "action_input_sequences.npy"
    action_targets_file = data_dir / "action_targets.npy"
    
    from setup_training import OSRSDataset
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
        batch_size=8,
        shuffle=True,
        device=device
    )
    
    # Create model and setup training components
    # Device already set above, reuse it
    model = setup_model(
        device=device,
        gamestate_dim=128,
        action_dim=8,
        sequence_length=10,
        hidden_dim=256,
        num_attention_heads=8
    )
    
    criterion, optimizer = setup_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=0.001,
        weight_decay=1e-4
    )
    
    # Move model to device
    model = model.to(device)
    
    print(f"Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Training configuration
    config = {
        'num_epochs': 10,
        'learning_rate': 0.001,
        'batch_size': train_loader.batch_size,
        'device': str(device)
    }
    
    # Train the model
    print("\nStarting training...")
    train_losses, val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=config['num_epochs'],
        device=device
    )
    
    # Save results
    print("\nSaving training results...")
    save_training_results(model, train_losses, val_losses, config)
    
    print("\nTraining completed successfully!")
    print(f"Final training loss: {train_losses[-1]:.4f}")
    print(f"Final validation loss: {val_losses[-1]:.4f}")
    print(f"Best validation loss: {min(val_losses):.4f}")
    
    if torch.cuda.is_available():
        print(f"Final CUDA memory: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB allocated, {torch.cuda.memory_reserved(0) / 1024**2:.1f} MB reserved")

if __name__ == "__main__":
    main()
