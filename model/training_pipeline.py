#!/usr/bin/env python3
"""
Training Pipeline for Imitation Learning Model
Curriculum Learning with Progressive Complexity
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from tqdm import tqdm

from imitation_hybrid_model import ImitationHybridModel
from action_tensor_loss import ActionTensorLoss

class OSRSDataset(Dataset):
    """Dataset for OSRS imitation learning"""
    
    def __init__(self, 
                 gamestate_features: np.ndarray,
                 action_targets: np.ndarray,
                 screenshots: Optional[np.ndarray] = None,
                 sequence_length: int = 10):
        self.gamestate_features = gamestate_features
        self.action_targets = action_targets
        self.screenshots = screenshots
        self.sequence_length = sequence_length
        
        # Calculate valid indices (need sequence_length gamestates)
        self.valid_indices = list(range(sequence_length - 1, len(gamestate_features)))
        
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        # Get the target index
        target_idx = self.valid_indices[idx]
        
        # Get temporal sequence (last sequence_length gamestates)
        start_idx = target_idx - self.sequence_length + 1
        temporal_sequence = self.gamestate_features[start_idx:target_idx + 1]
        
        # Current gamestate (last in sequence)
        current_gamestate = self.gamestate_features[target_idx]
        
        # Action target
        action_target = self.action_targets[target_idx]
        
        # Screenshot (if available)
        if self.screenshots is not None:
            screenshot = self.screenshots[target_idx]
        else:
            # Create dummy screenshot if not available
            screenshot = np.zeros((3, 224, 224), dtype=np.float32)
        
        # Convert to tensors
        current_gamestate = torch.FloatTensor(current_gamestate)
        temporal_sequence = torch.FloatTensor(temporal_sequence)
        screenshot = torch.FloatTensor(screenshot)
        action_target = torch.FloatTensor(action_target)
        
        return {
            'current_gamestate': current_gamestate,
            'temporal_sequence': temporal_sequence,
            'screenshot': screenshot,
            'action_target': action_target,
            'target_idx': target_idx
        }

class CurriculumTrainer:
    """Curriculum learning trainer for imitation learning"""
    
    def __init__(self, 
                 model: ImitationHybridModel,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 device: torch.device,
                 learning_rate: float = 0.001,
                 weight_decay: float = 1e-4):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Loss function
        self.criterion = ActionTensorLoss()
        
        # Optimizer
        self.optimizer = optim.Adam(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5
        )
        
        # Training history
        self.train_history = {
            'loss': [],
            'val_loss': [],
            'learning_rate': [],
            'epoch': []
        }
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_model_state = None
        
    def train_phase1(self, epochs: int = 20) -> Dict[str, List[float]]:
        """Phase 1: Train on simple actions (mouse movements only)"""
        print(f"🚀 Starting Phase 1: Mouse Movement Training ({epochs} epochs)")
        
        phase_history = {'loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            # Training
            train_loss = self._train_epoch_phase1()
            
            # Validation
            val_loss = self._validate_epoch_phase1()
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Store history
            phase_history['loss'].append(train_loss)
            phase_history['val_loss'].append(val_loss)
            
            # Print progress
            print(f"Phase 1, Epoch {epoch+1}/{epochs}: "
                  f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()
        
        return phase_history
    
    def train_phase2(self, epochs: int = 30) -> Dict[str, List[float]]:
        """Phase 2: Add clicks and basic interactions"""
        print(f"🚀 Starting Phase 2: Adding Click Training ({epochs} epochs)")
        
        phase_history = {'loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            # Training
            train_loss = self._train_epoch_phase2()
            
            # Validation
            val_loss = self._validate_epoch_phase2()
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Store history
            phase_history['loss'].append(train_loss)
            phase_history['val_loss'].append(val_loss)
            
            # Print progress
            print(f"Phase 2, Epoch {epoch+1}/{epochs}: "
                  f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()
        
        return phase_history
    
    def train_phase3(self, epochs: int = 50) -> Dict[str, List[float]]:
        """Phase 3: Full action space training"""
        print(f"🚀 Starting Phase 3: Full Action Training ({epochs} epochs)")
        
        phase_history = {'loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            # Training
            train_loss = self._train_epoch_phase3()
            
            # Validation
            val_loss = self._validate_epoch_phase3()
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Store history
            phase_history['loss'].append(train_loss)
            phase_history['val_loss'].append(val_loss)
            
            # Print progress
            print(f"Phase 3, Epoch {epoch+1}/{epochs}: "
                  f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.best_model_state = self.model.state_dict().copy()
        
        return phase_history
    
    def _train_epoch_phase1(self) -> float:
        """Train one epoch for Phase 1 (mouse movements only)"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in self.train_loader:
            # Move to device
            current_gamestate = batch['current_gamestate'].to(self.device)
            temporal_sequence = batch['temporal_sequence'].to(self.device)
            screenshot = batch['screenshot'].to(self.device)
            action_target = batch['action_target'].to(self.device)
            
            # Forward pass
            predictions = self.model(current_gamestate, screenshot, temporal_sequence)
            
            # Phase 1: Only mouse position loss
            loss = self.criterion(
                predictions={'mouse_position': predictions['mouse_position']},
                targets={'mouse_position': action_target[:, :2]}  # Only x, y coordinates
            )[0]
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def _train_epoch_phase2(self) -> float:
        """Train one epoch for Phase 2 (mouse movements + clicks)"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in self.train_loader:
            # Move to device
            current_gamestate = batch['current_gamestate'].to(self.device)
            temporal_sequence = batch['temporal_sequence'].to(self.device)
            screenshot = batch['screenshot'].to(self.device)
            action_target = batch['action_target'].to(self.device)
            
            # Forward pass
            predictions = self.model(current_gamestate, screenshot, temporal_sequence)
            
            # Phase 2: Mouse position + click loss
            loss = self.criterion(
                predictions={
                    'mouse_position': predictions['mouse_position'],
                    'mouse_click': predictions['mouse_click']
                },
                targets={
                    'mouse_position': action_target[:, :2],
                    'mouse_click': action_target[:, 2:4]  # left, right click
                }
            )[0]
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def _train_epoch_phase3(self) -> float:
        """Train one epoch for Phase 3 (full action space)"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in self.train_loader:
            # Move to device
            current_gamestate = batch['current_gamestate'].to(self.device)
            temporal_sequence = batch['temporal_sequence'].to(self.device)
            screenshot = batch['screenshot'].to(self.device)
            action_target = batch['action_target'].to(self.device)
            
            # Forward pass
            predictions = self.model(current_gamestate, screenshot, temporal_sequence)
            
            # Phase 3: Full action space loss
            loss = self.criterion(
                predictions=predictions,
                targets={
                    'mouse_position': action_target[:, :2],
                    'mouse_click': action_target[:, 2:4],
                    'key_press': action_target[:, 4:54],
                    'scroll': action_target[:, 54:56],
                    'confidence': action_target[:, 56:57],
                    'action_count': action_target[:, 57:73]
                }
            )[0]
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def _validate_epoch_phase1(self) -> float:
        """Validate one epoch for Phase 1"""
        return self._validate_epoch_generic(['mouse_position'])
    
    def _validate_epoch_phase2(self) -> float:
        """Validate one epoch for Phase 2"""
        return self._validate_epoch_generic(['mouse_position', 'mouse_click'])
    
    def _validate_epoch_phase3(self) -> float:
        """Validate one epoch for Phase 3"""
        return self._validate_epoch_generic(['mouse_position', 'mouse_click', 'key_press', 'scroll', 'confidence', 'action_count'])
    
    def _validate_epoch_generic(self, action_types: List[str]) -> float:
        """Generic validation for any phase"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move to device
                current_gamestate = batch['current_gamestate'].to(self.device)
                temporal_sequence = batch['temporal_sequence'].to(self.device)
                screenshot = batch['screenshot'].to(self.device)
                action_target = batch['action_target'].to(self.device)
                
                # Forward pass
                predictions = self.model(current_gamestate, screenshot, temporal_sequence)
                
                # Select relevant predictions and targets based on phase
                pred_dict = {}
                target_dict = {}
                
                if 'mouse_position' in action_types:
                    pred_dict['mouse_position'] = predictions['mouse_position']
                    target_dict['mouse_position'] = action_target[:, :2]
                
                if 'mouse_click' in action_types:
                    pred_dict['mouse_click'] = predictions['mouse_click']
                    target_dict['mouse_click'] = action_target[:, 2:4]
                
                if 'key_press' in action_types:
                    pred_dict['key_press'] = predictions['key_press']
                    target_dict['key_press'] = action_target[:, 4:54]
                
                if 'scroll' in action_types:
                    pred_dict['scroll'] = predictions['scroll']
                    target_dict['scroll'] = action_target[:, 54:56]
                
                if 'confidence' in action_types:
                    pred_dict['confidence'] = predictions['confidence']
                    target_dict['confidence'] = action_target[:, 56:57]
                
                if 'action_count' in action_types:
                    pred_dict['action_count'] = predictions['action_count']
                    target_dict['action_count'] = action_target[:, 57:73]
                
                # Compute loss
                loss = self.criterion(pred_dict, target_dict)[0]
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def train_with_curriculum(self, 
                             phase1_epochs: int = 20,
                             phase2_epochs: int = 30,
                             phase3_epochs: int = 50) -> Dict[str, List[float]]:
        """Complete curriculum training"""
        print("🎯 Starting Curriculum Learning Training...")
        
        # Phase 1: Mouse movements only
        phase1_history = self.train_phase1(phase1_epochs)
        
        # Phase 2: Add clicks
        phase2_history = self.train_phase2(phase2_epochs)
        
        # Phase 3: Full action space
        phase3_history = self.train_phase3(phase3_epochs)
        
        # Combine histories
        combined_history = {
            'loss': phase1_history['loss'] + phase2_history['loss'] + phase3_history['loss'],
            'val_loss': phase1_history['val_loss'] + phase2_history['val_loss'] + phase3_history['val_loss'],
            'phase_boundaries': [len(phase1_history['loss']), 
                               len(phase1_history['loss']) + len(phase2_history['loss'])]
        }
        
        # Save best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f"✅ Best model loaded (Val Loss: {self.best_val_loss:.4f})")
        
        return combined_history
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_history': self.train_history,
            'model_config': self.model.get_model_info()
        }, filepath)
        print(f"💾 Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_history = checkpoint['train_history']
        print(f"📂 Model loaded from {filepath}")
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot training history"""
        plt.figure(figsize=(12, 8))
        
        # Plot losses
        plt.subplot(2, 2, 1)
        epochs = range(1, len(self.train_history['loss']) + 1)
        plt.plot(epochs, self.train_history['loss'], 'b-', label='Training Loss')
        plt.plot(epochs, self.train_history['val_loss'], 'r-', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot learning rate
        plt.subplot(2, 2, 2)
        plt.plot(epochs, self.train_history['learning_rate'], 'g-')
        plt.title('Learning Rate')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.grid(True)
        
        # Plot loss breakdown (if available)
        if hasattr(self, 'loss_breakdown'):
            plt.subplot(2, 2, 3)
            loss_components = list(self.loss_breakdown.keys())
            loss_values = list(self.loss_breakdown.values())
            plt.bar(loss_components, loss_values)
            plt.title('Loss Components')
            plt.xlabel('Loss Type')
            plt.ylabel('Loss Value')
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Training history plot saved to {save_path}")
        
        plt.show()

def create_data_loaders(gamestate_features: np.ndarray,
                       action_targets: np.ndarray,
                       screenshots: Optional[np.ndarray] = None,
                       sequence_length: int = 10,
                       batch_size: int = 16,
                       train_split: float = 0.8,
                       random_seed: int = 42) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation data loaders"""
    
    # Set random seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    # Create dataset
    dataset = OSRSDataset(gamestate_features, action_targets, screenshots, sequence_length)
    
    # Split into train/val
    total_size = len(dataset)
    train_size = int(train_split * total_size)
    val_size = total_size - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0,  # Set to 0 for Windows compatibility
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0,  # Set to 0 for Windows compatibility
        pin_memory=True
    )
    
    print(f"📊 Data loaders created:")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {sequence_length}")
    
    return train_loader, val_loader

if __name__ == "__main__":
    print("Testing Training Pipeline...")
    
    # Create dummy data for testing
    num_samples = 100
    sequence_length = 10
    
    # Dummy gamestate features (128 features)
    gamestate_features = np.random.randn(num_samples, 128)
    
    # Dummy action targets (800 dimensions: 100 actions × 8 features)
    action_targets = np.random.randn(num_samples, 800)
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        gamestate_features, 
        action_targets, 
        sequence_length=sequence_length,
        batch_size=4
    )
    
    # Create model
    model = ImitationHybridModel(gamestate_dim=128)
    
    # Create trainer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = CurriculumTrainer(model, train_loader, val_loader, device)
    
    print(f"✅ Training pipeline created successfully!")
    print(f"Device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test a single training step
    print("\n🧪 Testing single training step...")
    batch = next(iter(train_loader))
    
    # Move to device
    current_gamestate = batch['current_gamestate'].to(device)
    temporal_sequence = batch['temporal_sequence'].to(device)
    screenshot = batch['screenshot'].to(device)
    action_target = batch['action_target'].to(device)
    
    # Forward pass
    with torch.no_grad():
        predictions = model(current_gamestate, screenshot, temporal_sequence)
    
    print(f"✅ Forward pass successful!")
    print(f"Input shapes: gamestate={current_gamestate.shape}, screenshot={screenshot.shape}, temporal={temporal_sequence.shape}")
    print(f"Output shapes:")
    for key, value in predictions.items():
        print(f"  {key}: {value.shape}")
    
    print(f"\n🎯 Ready for training!")










