import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import json
from typing import Tuple, Optional
import matplotlib.pyplot as plt
from datetime import datetime

# Import our feature extraction
from feature_extraction_rings import extract_ring_crafting_features, extract_action_features

class RingCraftingModel(nn.Module):
    """
    Neural network model for sapphire ring crafting.
    Takes game state features and predicts the next action.
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(RingCraftingModel, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Feature extraction layers
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Action prediction layers
        self.action_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: state -> action prediction
        
        Args:
            state: Game state features (batch_size, state_dim)
            
        Returns:
            action: Predicted action features (batch_size, action_dim)
        """
        # Encode state
        state_features = self.state_encoder(state)
        
        # Predict action
        action_pred = self.action_predictor(state_features)
        
        return action_pred

class RingCraftingDataset:
    """
    Dataset for ring crafting training data.
    Loads and preprocesses gamestates and actions.
    """
    
    def __init__(self, gamestates_dir: str, actions_file: str, sequence_length: int = 5):
        self.gamestates_dir = gamestates_dir
        self.actions_file = actions_file
        self.sequence_length = sequence_length
        
        # Load and process data
        self.states, self.actions = self._load_data()
        self.sequences = self._create_sequences()
        
        print(f"Dataset created: {len(self.sequences)} sequences")
    
    def _load_data(self) -> Tuple[list, list]:
        """Load gamestates and actions from files."""
        # Load gamestates
        gamestate_files = []
        for file in os.listdir(self.gamestates_dir):
            if file.endswith('.json'):
                gamestate_files.append(file)
        
        gamestate_files.sort()  # Sort by timestamp
        print(f"Found {len(gamestate_files)} gamestate files")
        
        states = []
        for file in gamestate_files:
            filepath = os.path.join(self.gamestates_dir, file)
            try:
                with open(filepath, 'r') as f:
                    gamestate = json.load(f)
                states.append(gamestate)
            except Exception as e:
                print(f"Error loading {file}: {e}")
                continue
        
        # Load actions (if available)
        actions = []
        if os.path.exists(self.actions_file):
            # TODO: Parse actions.csv and align with gamestates
            # For now, create dummy actions
            actions = [{'x': 0, 'y': 0, 'event_type': 'click', 'button': 'left'} for _ in states]
        else:
            # Create dummy actions for testing
            actions = [{'x': 0, 'y': 0, 'event_type': 'click', 'button': 'left'} for _ in states]
        
        return states, actions
    
    def _create_sequences(self) -> list:
        """Create training sequences from states and actions."""
        sequences = []
        
        for i in range(len(self.states) - self.sequence_length + 1):
            # Extract sequence
            state_seq = self.states[i:i + self.sequence_length]
            action_seq = self.actions[i:i + self.sequence_length]
            
            # Convert to features
            state_features = []
            action_features = []
            
            for state, action in zip(state_seq, action_seq):
                try:
                    state_feat = extract_ring_crafting_features(state)
                    action_feat = extract_action_features(action)
                    state_features.append(state_feat)
                    action_features.append(action_feat)
                except Exception as e:
                    print(f"Error extracting features: {e}")
                    continue
            
            if len(state_features) == self.sequence_length:
                sequences.append({
                    'states': np.array(state_features),
                    'actions': np.array(action_features)
                })
        
        return sequences
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a training sample."""
        seq = self.sequences[idx]
        
        # Convert to tensors
        states = torch.tensor(seq['states'], dtype=torch.float32)
        actions = torch.tensor(seq['actions'], dtype=torch.float32)
        
        return states, actions

def train_ring_bot(
    gamestates_dir: str,
    actions_file: str,
    model_save_path: str,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    sequence_length: int = 5
) -> RingCraftingModel:
    """
    Train the ring crafting bot model.
    
    Args:
        gamestates_dir: Directory containing gamestate JSON files
        actions_file: Path to actions CSV file
        model_save_path: Where to save the trained model
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate for optimizer
        sequence_length: Length of state-action sequences
        
    Returns:
        Trained model
    """
    
    print("🚀 Starting Ring Crafting Bot Training")
    print(f"📁 Gamestates: {gamestates_dir}")
    print(f"📁 Actions: {actions_file}")
    print(f"🔢 Sequence length: {sequence_length}")
    print(f"📊 Batch size: {batch_size}")
    print(f"🎯 Epochs: {epochs}")
    print(f"📚 Learning rate: {learning_rate}")
    
    # Create dataset
    dataset = RingCraftingDataset(gamestates_dir, actions_file, sequence_length)
    
    if len(dataset) == 0:
        raise ValueError("No training sequences found!")
    
    # Get feature dimensions from first sample
    sample_states, sample_actions = dataset[0]
    state_dim = sample_states.shape[-1]  # Last dimension of state features
    action_dim = sample_actions.shape[-1]  # Last dimension of action features
    
    print(f"📏 State features: {state_dim}")
    print(f"📏 Action features: {action_dim}")
    
    # Create model
    model = RingCraftingModel(state_dim, action_dim)
    print(f"🤖 Model created: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    # Training loop
    model.train()
    train_losses = []
    
    print("\n🎯 Starting training...")
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        # Process sequences in batches
        for i in range(0, len(dataset), batch_size):
            batch_end = min(i + batch_size, len(dataset))
            batch_states = []
            batch_actions = []
            
            # Collect batch data
            for j in range(i, batch_end):
                states, actions = dataset[j]
                batch_states.append(states)
                batch_actions.append(actions)
            
            if not batch_states:
                continue
            
            # Convert to tensors
            batch_states = torch.stack(batch_states)  # (batch_size, seq_len, state_dim)
            batch_actions = torch.stack(batch_actions)  # (batch_size, seq_len, action_dim)
            
            # Forward pass
            optimizer.zero_grad()
            
            # Predict actions for each state in sequence
            total_loss = 0
            for seq_idx in range(sequence_length):
                state_input = batch_states[:, seq_idx, :]  # (batch_size, state_dim)
                target_action = batch_actions[:, seq_idx, :]  # (batch_size, action_dim)
                
                predicted_action = model(state_input)
                loss = criterion(predicted_action, target_action)
                total_loss += loss
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
            num_batches += 1
        
        # Calculate average loss
        if num_batches > 0:
            avg_loss = epoch_loss / num_batches
            train_losses.append(avg_loss)
            
            # Update learning rate
            scheduler.step(avg_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Print progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:3d}/{epochs} | Loss: {avg_loss:.6f} | LR: {current_lr:.2e}")
        
        # Save checkpoint every 25 epochs
        if (epoch + 1) % 25 == 0:
            checkpoint_path = model_save_path.replace('.pth', f'_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'train_losses': train_losses
            }, checkpoint_path)
            print(f"💾 Checkpoint saved: {checkpoint_path}")
    
    # Save final model
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
        'train_losses': train_losses,
        'state_dim': state_dim,
        'action_dim': action_dim,
        'sequence_length': sequence_length
    }, model_save_path)
    
    print(f"\n✅ Training complete! Model saved to: {model_save_path}")
    
    # Plot training loss
    if len(train_losses) > 1:
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses)
        plt.title('Training Loss Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        
        plot_path = model_save_path.replace('.pth', '_training_loss.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"📊 Training loss plot saved to: {plot_path}")
    
    return model

def evaluate_model(model: RingCraftingModel, test_dataset: RingCraftingDataset) -> dict:
    """
    Evaluate the trained model on test data.
    
    Args:
        model: Trained model
        test_dataset: Test dataset
        
    Returns:
        Dictionary containing evaluation metrics
    """
    model.eval()
    
    total_loss = 0.0
    num_samples = 0
    
    with torch.no_grad():
        for states, actions in test_dataset:
            # Predict actions for each state in sequence
            for seq_idx in range(states.shape[0]):
                state_input = states[seq_idx:seq_idx+1, :]  # Add batch dimension
                target_action = actions[seq_idx:seq_idx+1, :]
                
                predicted_action = model(state_input)
                loss = nn.MSELoss()(predicted_action, target_action)
                total_loss += loss.item()
                num_samples += 1
    
    avg_loss = total_loss / num_samples if num_samples > 0 else float('inf')
    
    return {
        'test_loss': avg_loss,
        'num_samples': num_samples
    }

if __name__ == "__main__":
    # Configuration
    GAMESTATES_DIR = "data/gamestates"
    ACTIONS_FILE = "data/actions.csv"  # Will be created when you record actions
    MODEL_SAVE_PATH = "../models/ring_crafting_bot.pth"
    
    # Training parameters
    EPOCHS = 100
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    SEQUENCE_LENGTH = 5
    
    # Check if gamestates directory exists
    if not os.path.exists(GAMESTATES_DIR):
        print(f"❌ Gamestates directory not found: {GAMESTATES_DIR}")
        print("Please run the data collection first!")
        exit(1)
    
    # Create models directory
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    
    try:
        # Train the model
        model = train_ring_bot(
            gamestates_dir=GAMESTATES_DIR,
            actions_file=ACTIONS_FILE,
            model_save_path=MODEL_SAVE_PATH,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            sequence_length=SEQUENCE_LENGTH
        )
        
        print("\n🎉 Ring crafting bot training completed successfully!")
        print(f"🤖 Model saved to: {MODEL_SAVE_PATH}")
        print("\n📋 Next steps:")
        print("1. Collect more training data by recording your crafting sessions")
        print("2. Retrain the model with more data")
        print("3. Create the bot execution system")
        print("4. Test the bot in-game")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
