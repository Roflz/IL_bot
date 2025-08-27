#!/usr/bin/env python3
"""
OSRS Imitation Learning Training Setup Script
Compatible with the user's data format:
- Gamestate sequences: (180, 10, 128) - 180 sequences, 10 timesteps, 128 features
- Action input sequences: 180 sequences of 10 timesteps each
- Action targets: 180 sequences with variable action counts
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import time
import os

from model.imitation_hybrid_model import ImitationHybridModel
from model.action_tensor_loss import ActionTensorLoss

class OSRSDataset(Dataset):
    """
    Dataset for OSRS training data compatible with the user's format.
    
    Data structure:
    - Input: 10 timesteps of gamestate features (128 features each)
    - Target: Variable-length action sequences (8 features per action)
    """
    
    def __init__(self, 
                 gamestate_file: str,
                 action_input_file: str,
                 action_targets_file: str,
                 sequence_length: int = 10,
                 max_actions: int = 100):
        
        # Load data files
        self.gamestate_sequences = np.load(gamestate_file)  # (batch_size, 10, 128)
        
        # Load action input sequences as numpy array
        self.action_input_sequences = np.load(action_input_file)  # (batch_size, 10, 100, 8)
        
        # Load action targets as numpy array  
        self.action_targets = np.load(action_targets_file)  # (batch_size, 100, 8)
        
        self.sequence_length = sequence_length
        self.max_actions = max_actions
        self.n_sequences = len(self.gamestate_sequences)
        
        print(f"Dataset loaded successfully!")
        print(f"  Gamestate sequences: {self.gamestate_sequences.shape}")
        print(f"  Action input sequences: {self.action_input_sequences.shape}")
        print(f"  Action targets: {self.action_targets.shape}")
        print(f"  Max actions per sequence: {max_actions}")
        
        # Validate data compatibility
        assert len(self.gamestate_sequences) == len(self.action_input_sequences) == len(self.action_targets), \
            "All data files must have the same number of sequences"
        
        assert self.gamestate_sequences.shape[1] == sequence_length, \
            f"Gamestate sequences must have {sequence_length} timesteps"
        
        assert self.gamestate_sequences.shape[2] == 128, \
            "Gamestate sequences must have 128 features"
        
        # Validate action sequence shapes
        assert self.action_input_sequences.shape[1] == sequence_length, \
            f"Action input sequences must have {sequence_length} timesteps"
        
        assert self.action_input_sequences.shape[2] == max_actions, \
            f"Action input sequences must have {max_actions} actions per timestep"
        
        assert self.action_input_sequences.shape[3] == 8, \
            "Action input sequences must have 8 features per action"
        
        assert self.action_targets.shape[1] == max_actions, \
            f"Action targets must have {max_actions} actions"
        
        assert self.action_targets.shape[2] == 8, \
            "Action targets must have 8 features per action"
    
    def __len__(self):
        return self.n_sequences
    
    def __getitem__(self, idx):
        # Get gamestate sequence (10 timesteps, 128 features)
        gamestate_sequence = self.gamestate_sequences[idx]  # (10, 128)
        
        # Get temporal sequence (all 10 timesteps)
        temporal_sequence = gamestate_sequence  # (10, 128)
        
        # Get action input sequence (10 timesteps, 100 actions, 8 features)
        action_sequence = self.action_input_sequences[idx]  # (10, 100, 8)
        
        # Get action target for this sequence
        action_target = self.action_targets[idx]  # (100, 8)
        
        return {
            'temporal_sequence': torch.FloatTensor(temporal_sequence),
            'action_sequence': torch.FloatTensor(action_sequence),
            'action_target': torch.FloatTensor(action_target)
        }
    
    # Note: Old parsing methods removed since data is now loaded as numpy arrays

def optimize_batch_size_for_cuda(device: torch.device, base_batch_size: int = 8) -> int:
    """Optimize batch size based on available CUDA memory"""
    if not torch.cuda.is_available():
        return base_batch_size
    
    # Get available CUDA memory
    total_memory = torch.cuda.get_device_properties(0).total_memory
    available_memory = total_memory * 0.8  # Use 80% of available memory
    
    # Estimate memory per sample (rough estimate for the model)
    # Model has ~5.7M parameters, each sample is ~10 timesteps × 128 features + actions
    estimated_memory_per_sample = 50 * 1024 * 1024  # 50 MB per sample (conservative estimate)
    
    # Calculate optimal batch size
    optimal_batch_size = int(available_memory / estimated_memory_per_sample)
    
    # Ensure batch size is reasonable
    optimal_batch_size = max(1, min(optimal_batch_size, base_batch_size * 4))
    
    print(f"CUDA memory optimization:")
    print(f"  Total memory: {total_memory / 1024**3:.1f} GB")
    print(f"  Available memory: {available_memory / 1024**3:.1f} GB")
    print(f"  Estimated memory per sample: {estimated_memory_per_sample / 1024**2:.1f} MB")
    print(f"  Optimal batch size: {optimal_batch_size}")
    
    return optimal_batch_size

def create_data_loaders(dataset: OSRSDataset, 
                       train_split: float = 0.8,
                       batch_size: int = 8,
                       shuffle: bool = True,
                       device: torch.device = None) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation data loaders"""
    
    # Optimize batch size for CUDA if available
    if device and torch.cuda.is_available():
        batch_size = optimize_batch_size_for_cuda(device, batch_size)
    
    # Split dataset
    n_train = int(len(dataset) * train_split)
    n_val = len(dataset) - n_train
    
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [n_train, n_val])
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        num_workers=0,  # Set to 0 for Windows compatibility
        pin_memory=torch.cuda.is_available()  # Enable pin_memory for CUDA
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()  # Enable pin_memory for CUDA
    )
    
    print(f"Data loaders created:")
    print(f"  Train: {len(train_loader)} batches ({len(train_dataset)} samples)")
    print(f"  Validation: {len(val_loader)} batches ({len(val_dataset)} samples)")
    print(f"  Batch size: {batch_size}")
    
    return train_loader, val_loader

def setup_model(device: torch.device, 
                gamestate_dim: int = 128,
                action_dim: int = 8,
                sequence_length: int = 10,
                hidden_dim: int = 256,
                num_attention_heads: int = 8) -> ImitationHybridModel:
    """Create and setup the imitation learning model"""
    
    print(f"Setting up model...")
    print(f"  Gamestate features: {gamestate_dim}")
    print(f"  Action features: {action_dim}")
    print(f"  Sequence length: {sequence_length}")
    print(f"  Hidden dimension: {hidden_dim}")
    print(f"  Attention heads: {num_attention_heads}")
    
    # Create model
    model = ImitationHybridModel(
        gamestate_dim=gamestate_dim,
        action_dim=action_dim,
        sequence_length=sequence_length,
        hidden_dim=hidden_dim,
        num_attention_heads=num_attention_heads
    )
    
    # Move to device
    model = model.to(device)
    
    # Get model info
    model_info = model.get_model_info()
    print(f"  Model created successfully!")
    print(f"  Total parameters: {model_info['total_parameters']:,}")
    print(f"  Trainable parameters: {model_info['trainable_parameters']:,}")
    
    return model

def setup_training(model: ImitationHybridModel,
                  train_loader: DataLoader,
                  val_loader: DataLoader,
                  device: torch.device,
                  learning_rate: float = 3e-4,
                  weight_decay: float = 1e-4) -> Tuple[ActionTensorLoss, optim.Optimizer]:
    """Setup loss function and optimizer"""
    
    print(f"Setting up training components...")
    
    # Loss function
    criterion = ActionTensorLoss()
    print(f"  Loss function: {criterion.__class__.__name__}")
    
    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    print(f"  Optimizer: Adam (lr={learning_rate}, weight_decay={weight_decay})")
    
    return criterion, optimizer

def test_model_compatibility(model: ImitationHybridModel, 
                           train_loader: DataLoader,
                           device: torch.device) -> bool:
    """Test if the model can process a batch of data"""
    
    print(f"Testing model compatibility...")
    
    try:
        # Get a batch
        batch = next(iter(train_loader))
        
        # Extract inputs
        temporal_sequence = batch['temporal_sequence'].to(device)
        action_sequence = batch['action_sequence'].to(device)
        
        print(f"  Input shapes:")
        print(f"    Temporal sequence: {temporal_sequence.shape}")
        print(f"    Action sequence: {action_sequence.shape}")
        
        # Test forward pass
        with torch.no_grad():
            output = model(temporal_sequence, action_sequence)
        
        print(f"  Output shape: {output.shape}")
        
        print(f"  ✓ Model compatibility test passed!")
        
        # CUDA memory info if available
        if torch.cuda.is_available():
            print(f"  CUDA memory after test: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB allocated")
            torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"  ✗ Model compatibility test failed: {e}")
        return False

def optimize_cuda_settings():
    """Optimize CUDA settings for better performance"""
    if torch.cuda.is_available():
        # Enable cuDNN benchmarking for faster convolutions
        torch.backends.cudnn.benchmark = True
        # Enable cuDNN deterministic mode for reproducibility
        torch.backends.cudnn.deterministic = False
        print("CUDA optimizations enabled")

def main():
    """Main setup function"""
    print("OSRS Imitation Learning Training Setup")
    print("=" * 60)
    
    # Optimize CUDA settings
    optimize_cuda_settings()
    
    # Check CUDA availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"CUDA compute capability: {torch.cuda.get_device_capability(0)}")
    else:
        print("CUDA not available, using CPU")
    print(f"Device: {device}")
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test OSRS Training Setup')
    parser.add_argument('--data_dir', type=str, default="data/06_final_training_data",
                       help='Path to training data directory (default: data/06_final_training_data)')
    args = parser.parse_args()
    
    # Data file paths
    data_dir = Path(args.data_dir)
    gamestate_file = data_dir / "gamestate_sequences.npy"
    action_input_file = data_dir / "action_input_sequences.npy"
    action_targets_file = data_dir / "action_targets.npy"
    
    # Check if files exist
    for file_path in [gamestate_file, action_input_file, action_targets_file]:
        if not file_path.exists():
            print(f"ERROR: Required file not found: {file_path}")
            return
    
    print(f"Data files found successfully!")
    
    # Create dataset
    print(f"\nCreating dataset...")
    dataset = OSRSDataset(
        gamestate_file=str(gamestate_file),
        action_input_file=str(action_input_file),
        action_targets_file=str(action_targets_file),
        sequence_length=10,
        max_actions=100
    )
    
    # Create data loaders
    print(f"\nCreating data loaders...")
    train_loader, val_loader = create_data_loaders(
        dataset=dataset,
        train_split=0.8,
        batch_size=8,
        shuffle=True,
        device=device
    )
    
    # Setup model
    print(f"\nSetting up model...")
    model = setup_model(
        device=device,
        gamestate_dim=128,
        action_dim=8,
        sequence_length=10,
        hidden_dim=256,
        num_attention_heads=8
    )
    
    # Setup training components
    print(f"\nSetting up training components...")
    criterion, optimizer = setup_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=0.001,
        weight_decay=1e-4
    )
    
    # Test model compatibility
    print(f"\nTesting model compatibility...")
    compatibility_ok = test_model_compatibility(model, train_loader, device)
    
    if compatibility_ok:
        print(f"\n{'='*60}")
        print("SETUP COMPLETED SUCCESSFULLY!")
        print(f"{'='*60}")
        print("Your model is ready for training!")
        print(f"\nNext steps:")
        print(f"1. Run the training script")
        print(f"2. Monitor training progress")
        print(f"3. Evaluate model performance")
        print(f"\nTraining configuration:")
        print(f"  Model: ImitationHybridModel")
        print(f"  Dataset: {len(dataset)} sequences")
        print(f"  Batch size: {train_loader.batch_size}")
        print(f"  Learning rate: 0.001")
        print(f"  Device: {device}")
        
        # Save setup info
        setup_info = {
            'model_type': 'ImitationHybridModel',
            'dataset_size': len(dataset),
            'train_size': len(train_loader.dataset),
            'val_size': len(val_loader.dataset),
            'batch_size': train_loader.batch_size,
            'learning_rate': 0.001,
            'weight_decay': 1e-4,
            'device': str(device),
            'model_parameters': model.get_model_info()
        }
        
        with open('training_setup_info.json', 'w') as f:
            json.dump(setup_info, f, indent=2)
        
        print(f"\nSetup info saved to: training_setup_info.json")
        
    else:
        print(f"\n{'='*60}")
        print("SETUP FAILED!")
        print(f"{'='*60}")
        print("Please check the error messages above and fix any issues.")

if __name__ == "__main__":
    main()
