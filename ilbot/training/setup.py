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


from ilbot.model.advanced_losses import AdvancedUnifiedEventLoss
from ilbot.utils.normalization import normalize_gamestate_features, normalize_action_input_features, normalize_action_target_features

class OSRSDataset(Dataset):
    """
    Loads training data from data_dir.
    Uses current file format: action_targets.npy + valid_mask.npy
    Shapes (T, G, A, Fin) are inferred from files – no constants.
    """
    def __init__(self, data_dir: Union[str, Path],
                 use_log1p_time: bool = True,
                 time_div_ms: int = 1000,
                 time_clip_s: Optional[float] = None,
                 enum_sizes: Optional[Dict[str, int]] = None,
                 device: Optional[torch.device] = None):
        self.data_dir = Path(data_dir)
        self.manifest = self._load_manifest(self.data_dir)
        # Enums from manifest or explicit arg
        self.enums = enum_sizes or self.manifest.get("enums", {})
        
        # Load core tensors - current format only
        self.gamestate_sequences = np.load(self.data_dir/"gamestate_sequences.npy")      # (B, T, G)
        self.action_input_sequences = np.load(self.data_dir/"action_input_sequences.npy")# (B, T, A, Fin)
        self.action_targets = np.load(self.data_dir/"action_targets.npy")                # (B, A, 7)
        
        # Create valid_mask from action_targets (non-zero actions are valid)
        self.valid_mask = (np.abs(self.action_targets).sum(axis=-1) > 0)  # (B, A)

        self.B, self.T, self.G = self.gamestate_sequences.shape
        _, _, self.A, self.Fin = self.action_input_sequences.shape
        self.n_sequences = self.B
        print("Dataset loaded successfully!")
        print(f"  Gamestate sequences: {self.gamestate_sequences.shape}")
        print(f"  Action input sequences: {self.action_input_sequences.shape}")
        print(f"  Action targets: {self.action_targets.shape}")
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




