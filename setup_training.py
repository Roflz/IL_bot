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
        self.gamestate_sequences = np.load(gamestate_file)  # (180, 10, 128)
        
        with open(action_input_file, 'r') as f:
            self.action_input_sequences = json.load(f)  # 180 sequences
        
        with open(action_targets_file, 'r') as f:
            self.action_targets = json.load(f)  # 180 sequences
        
        self.sequence_length = sequence_length
        self.max_actions = max_actions
        self.n_sequences = len(self.gamestate_sequences)
        
        print(f"Dataset loaded successfully!")
        print(f"  Gamestate sequences: {self.gamestate_sequences.shape}")
        print(f"  Action input sequences: {len(self.action_input_sequences)}")
        print(f"  Action targets: {len(self.action_targets)}")
        print(f"  Max actions per sequence: {max_actions}")
        
        # Validate data compatibility
        assert len(self.gamestate_sequences) == len(self.action_input_sequences) == len(self.action_targets), \
            "All data files must have the same number of sequences"
        
        assert self.gamestate_sequences.shape[1] == sequence_length, \
            f"Gamestate sequences must have {sequence_length} timesteps"
        
        assert self.gamestate_sequences.shape[2] == 128, \
            "Gamestate sequences must have 128 features"
    
    def __len__(self):
        return self.n_sequences
    
    def __getitem__(self, idx):
        # Get gamestate sequence (10 timesteps, 128 features)
        gamestate_sequence = self.gamestate_sequences[idx]  # (10, 128)
        
        # Get temporal sequence (all 10 timesteps)
        temporal_sequence = gamestate_sequence  # (10, 128)
        
        # Get action input sequence (10 timesteps, 8 features)
        action_sequence = self._parse_action_input_sequence(self.action_input_sequences[idx])
        
        # Get action target for this sequence
        action_target = self._parse_action_target(self.action_targets[idx])
        
        return {
            'temporal_sequence': torch.FloatTensor(temporal_sequence),
            'action_sequence': torch.FloatTensor(action_sequence),
            'action_target': action_target
        }
    
    def _parse_action_target(self, action_sequence: List) -> torch.Tensor:
        """
        Parse action target sequence into model-compatible format.
        
        Args:
            action_sequence: List like [action_count, timing1, type1, x1, y1, button1, key1, scroll1, ...]
            
        Returns:
            Action tensor with count at index 0: (max_actions + 1, 8)
        """
        if len(action_sequence) == 0:
            # No actions
            action_tensor = torch.zeros(self.max_actions + 1, 8)
            action_tensor[0, 0] = 0  # Action count at index 0
            return action_tensor
        
        # First element is action count
        action_count = int(action_sequence[0])
        
        # Create action tensor (max_actions + 1, 8) - +1 for action count at index 0
        action_tensor = torch.zeros(self.max_actions + 1, 8)
        
        # Set action count at index 0: [action_count, 0, 0, 0, 0, 0, 0, 0]
        action_tensor[0, 0] = action_count
        
        # Parse the flat list into 8-feature actions
        # Format: [action_count, timing1, type1, x1, y1, button1, key1, scroll1, timing2, type2, x2, y2, button2, key2, scroll2, ...]
        features_per_action = 8
        actions_start_idx = 1  # Skip action_count
        
        # Each action has 8 features: [timing, type, x, y, button, key, scroll_dx, scroll_dy]
        max_actions_found = min((len(action_sequence) - actions_start_idx) // features_per_action, self.max_actions)
        
        for i in range(max_actions_found):
            start_idx = actions_start_idx + i * features_per_action
            end_idx = start_idx + features_per_action
            
            if end_idx <= len(action_sequence):
                action_features = action_sequence[start_idx:end_idx]
                action_tensor[i + 1] = torch.FloatTensor(action_features)  # +1 because index 0 is count
        
        return action_tensor  # Shape: (max_actions + 1, 8)
    
    def _parse_action_input_sequence(self, action_sequence: List) -> np.ndarray:
        """
        Parse action input sequence into proper tensor format.
        
        Args:
            action_sequence: List of 10 timesteps, each with action data
            
        Returns:
            Numpy array of shape (10, 100, 8) - 10 timesteps, each with up to 100 actions, 8 features per action
        """
        # action_sequence is a list of 10 timesteps
        # Each timestep is: [action_count, action1_feats, action2_feats, ...]
        
        timesteps = 10
        max_actions_per_timestep = 100
        features_per_action = 8
        
        # Output: (10, 101, 8) - 10 timesteps, each with action count at index 0 + up to 100 actions
        action_array = np.zeros((timesteps, max_actions_per_timestep + 1, features_per_action), dtype=np.float32)
        
        # Parse each of the 10 timesteps
        for i in range(min(timesteps, len(action_sequence))):
            timestep_data = action_sequence[i]  # One timestep's data
            
            if len(timestep_data) >= 1:
                action_count = int(timestep_data[0])
                actions_start_idx = 1
                
                # Store action count in the first action's first position (like targets do)
                action_array[i, 0, 0] = action_count
                
                # Parse remaining actions (starting from index 1)
                max_actions_found = min((len(timestep_data) - actions_start_idx) // features_per_action, max_actions_per_timestep)
                
                for j in range(max_actions_found):
                    start_idx = actions_start_idx + j * features_per_action
                    end_idx = start_idx + features_per_action
                    if end_idx <= len(timestep_data):
                        action_features = timestep_data[start_idx:end_idx]
                        action_array[i, j + 1] = action_features  # +1 because index 0 is for count
        
        return action_array  # Shape: (10, 100, 8)

def create_data_loaders(dataset: OSRSDataset, 
                       train_split: float = 0.8,
                       batch_size: int = 8,
                       shuffle: bool = True) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation data loaders"""
    
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
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"Data loaders created:")
    print(f"  Train: {len(train_loader)} batches ({len(train_dataset)} samples)")
    print(f"  Validation: {len(val_loader)} batches ({len(val_dataset)} samples)")
    
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
                  learning_rate: float = 0.001,
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
        return True
        
    except Exception as e:
        print(f"  ✗ Model compatibility test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("OSRS Imitation Learning Training Setup")
    print("=" * 60)
    
    # Check CUDA availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data file paths
    data_dir = Path("data/final_training_data")
    gamestate_file = data_dir / "gamestate_sequences.npy"
    action_input_file = data_dir / "action_input_sequences.json"
    action_targets_file = data_dir / "action_targets.json"
    
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
        shuffle=True
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
        print(f"  Batch size: 8")
        print(f"  Learning rate: 0.001")
        print(f"  Device: {device}")
        
        # Save setup info
        setup_info = {
            'model_type': 'ImitationHybridModel',
            'dataset_size': len(dataset),
            'train_size': len(train_loader.dataset),
            'val_size': len(val_loader.dataset),
            'batch_size': 8,
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
