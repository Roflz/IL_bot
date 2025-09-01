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
from typing import Dict, List, Tuple, Optional, Union, Any
import os

from ilbot.model.imitation_hybrid_model import ImitationHybridModel
from ilbot.model.advanced_losses import AdvancedUnifiedEventLoss
from ilbot.utils.normalization import normalize_gamestate_features, normalize_action_input_features, normalize_action_target_features

class OSRSDataset(Dataset):
    """
    Loads V1 or V2 targets automatically from a data_dir.
    - V1: action_targets.npy  (B, A, 8)
    - V2: actions_v2.npy      (B, A, 7)  + valid_mask.npy (B, A)
    Shapes (T, G, A, Fin) are inferred from files – no constants.
    """
    def __init__(self, data_dir: Union[str, Path],
                 targets_version: Optional[str] = None,
                 use_log1p_time: bool = True,
                 time_div_ms: int = 1000,
                 time_clip_s: Optional[float] = None,
                 enum_sizes: Optional[Dict[str, int]] = None,
                 device: Optional[torch.device] = None):
        self.data_dir = Path(data_dir)
        self.manifest = self._load_manifest(self.data_dir)
        # Prefer explicit arg, then manifest, then default to "v1"
        self.targets_version = targets_version or self.manifest.get("targets_version", "v1")
        # Enums (for v2 heads); prefer explicit arg, else manifest, else empty
        self.enums = enum_sizes or self.manifest.get("enums", {})
        
        # Load data based on targets version
        self.gamestate_sequences = np.load(self.data_dir/"gamestate_sequences.npy")      # (B, T, G)
        self.action_input_sequences = np.load(self.data_dir/"action_input_sequences.npy")# (B, T, A, Fin)
        
        # V2 tensors (present only when targets_version == "v2")
        self.actions_v2 = None
        if self.targets_version == "v2":
            self.actions_v2 = np.load(self.data_dir/"actions_v2.npy")
            self.valid_mask = np.load(self.data_dir/"valid_mask.npy").astype(bool)
            self.action_targets = self.actions_v2
        else:
            # V1 fallback: keep current loading path
            self.action_targets = np.load(self.data_dir/"action_targets.npy")
            # infer mask from all-zero rows (padding)
            self.valid_mask = (np.abs(self.action_targets).sum(axis=-1) > 0)

        self.B, self.T, self.G = self.gamestate_sequences.shape
        _, _, self.A, self.Fin = self.action_input_sequences.shape
        self.n_sequences = self.B
        print("Dataset loaded successfully!")
        print(f"  Gamestate sequences: {self.gamestate_sequences.shape}")
        print(f"  Action input sequences: {self.action_input_sequences.shape}")
        print(f"  Action targets: {self.action_targets.shape} ({self.targets_version})")
        print(f"  Max actions per sequence: {self.A}")

    def _load_manifest(self, root: Path) -> Dict[str, Any]:
        manifest_path = root / "dataset_manifest.json"
        if manifest_path.exists():
            with open(manifest_path, "r") as f:
                return json.load(f)
        return {}

    def get_enums(self) -> Dict[str, int]:
        """Return enum sizes for v2 (button, key_action, key_id, scroll_y)."""
        return self.enums

    def __len__(self): return self.n_sequences

    def __getitem__(self, idx):
        # Load raw data
        temporal_sequence = torch.from_numpy(self.gamestate_sequences[idx]).float()   # (T,G)
        action_sequence = torch.from_numpy(self.action_input_sequences[idx]).float()  # (T,A,Fin)
        action_target = torch.from_numpy(self.action_targets[idx]).float()            # (A,7|8)
        valid_mask = torch.from_numpy(self.valid_mask[idx]).bool()                    # (A,)
        
        # Normalize data using shared normalization utilities
        temporal_sequence_normalized = normalize_gamestate_features(temporal_sequence)
        action_sequence_normalized = normalize_action_input_features(action_sequence)
        action_target_normalized = normalize_action_target_features(action_target)
        
        return {
            "temporal_sequence": temporal_sequence_normalized,
            "action_sequence": action_sequence_normalized,
            "action_target": action_target_normalized,
            "valid_mask": valid_mask,
            "targets_version": self.targets_version,
            "manifest": self.manifest or {},
            "screen_dimensions": (1920.0, 1080.0)  # Store for denormalization
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

def print_cuda_info():
    print("CUDA optimizations enabled")
    if torch.cuda.is_available():
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"CUDA compute capability: {torch.cuda.get_device_capability(0)}")

def create_data_loaders(
    dataset: OSRSDataset,
    train_split=0.8,
    batch_size=32,
    shuffle=True,
    device=None,
    disable_cuda_batch_opt: bool = False
):
    """Create train and validation data loaders"""
    
    # Optional: auto-optimize batch size for available CUDA memory
    if device and device.type == 'cuda' and not disable_cuda_batch_opt:
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
    
    print(f"Data loaders created:\n  Train: {len(train_loader)} batches ({len(train_dataset)} samples)\n  Validation: {len(val_loader)} batches ({len(val_dataset)} samples)\n  Batch size: {batch_size}")
    
    return train_loader, val_loader

def setup_model(device: torch.device,
                *,
                gamestate_dim: int,
                action_dim: int,
                sequence_length: int,
                hidden_dim: int = 256,
                num_attention_heads: int = 8,
                head_version: str = "v1",
                enum_sizes: Optional[Dict[str, int]] = None) -> ImitationHybridModel:
    """Create and setup the imitation learning model"""
    
    print(f"Setting up model...")
    print(f"  Gamestate features: {gamestate_dim}")
    print(f"  Action features: {action_dim}")
    print(f"  Sequence length: {sequence_length}")
    print(f"  Hidden dimension: {hidden_dim}")
    print(f"  Attention heads: {num_attention_heads}")
    
    # Create data_config dict for the model
    data_config = {
        'gamestate_dim': gamestate_dim,
        'max_actions': 100,  # Fixed for this model
        'action_features': action_dim,
        'temporal_window': sequence_length,
        'enum_sizes': enum_sizes or {},
        'event_types': 4  # Fixed for this model
    }
    
    # Create model
    model = ImitationHybridModel(
        data_config=data_config,
        hidden_dim=hidden_dim,
        num_heads=num_attention_heads,
        num_layers=6  # Default value
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
                  weight_decay: float = 1e-4) -> Tuple[AdvancedUnifiedEventLoss, optim.Optimizer]:
    """Setup loss function and optimizer"""
    
    print(f"Setting up training components...")
    
    # Loss function
    criterion = AdvancedUnifiedEventLoss()
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


