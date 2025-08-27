import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from data_loader import ILSequenceDataset
from model import ImitationHybridModel
import numpy as np
import os

# Config
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/aligned'))
SEQ_LEN = 10
BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATE = 1e-4

# Dataset and DataLoader
train_dataset = ILSequenceDataset(DATA_DIR, seq_len=SEQ_LEN, use_precomputed_features=True)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Dynamically determine feature dimensions
state_features_path = os.path.join(DATA_DIR, 'state_features.npy')
action_features_path = os.path.join(DATA_DIR, 'action_features.npy')
state_features = np.load(state_features_path)
action_features = np.load(action_features_path)
STRUCTURED_INPUT_DIM = state_features.shape[1]
ACTION_OUTPUT_DIM = action_features.shape[1]
print(f"Detected STRUCTURED_INPUT_DIM: {STRUCTURED_INPUT_DIM}")
print(f"Detected ACTION_OUTPUT_DIM: {ACTION_OUTPUT_DIM}")

del state_features, action_features  # free memory

# Model
model = ImitationHybridModel(STRUCTURED_INPUT_DIM, ACTION_OUTPUT_DIM)
model.train()

# Loss and optimizer
criterion = nn.MSELoss()  # Placeholder: use appropriate loss for your action space
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
for epoch in range(EPOCHS):
    total_loss = 0.0
    for batch_idx, (images, states, actions) in enumerate(train_loader):
        # images: (B, T, C, H, W), states: (B, T, F), actions: (B, T, A)
        optimizer.zero_grad()
        preds = model(images, states)  # (B, T, A)
        loss = criterion(preds, actions)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if (batch_idx + 1) % 10 == 0:
            print(f'Epoch {epoch+1} Batch {batch_idx+1} Loss: {loss.item():.4f}')
    avg_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch+1} Average Loss: {avg_loss:.4f}')

# TODO: Add validation, checkpointing, and custom loss for classification/discrete actions
# TODO: Add evaluation metrics and logging
# TODO: Save model after training 