import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import json
from .feature_extraction import extract_state_features, extract_action_features

class ILSequenceDataset(Dataset):
    """
    PyTorch Dataset for OSRS imitation learning.
    Loads aligned (screenshot, structured state, action) sequences.
    Allows configurable number of closest NPCs, objects, widgets for feature extraction.
    Supports loading from precomputed .npy feature arrays for efficiency.
    """
    def __init__(self, data_dir, seq_len=10, n_npcs=10, n_objects=10, n_widgets=10, transform=None, use_precomputed_features=False):
        """
        Args:
            data_dir: Directory with aligned data (expects subdirs: screenshots/, states.json, actions.json, screenshot_paths.json, state_features.npy, action_features.npy)
            seq_len: Length of each sequence window
            n_npcs, n_objects, n_widgets: number of closest entities to include
            transform: torchvision transforms for images
            use_precomputed_features: If True, load from .npy arrays and screenshot_paths.json
        """
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.n_npcs = n_npcs
        self.n_objects = n_objects
        self.n_widgets = n_widgets
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.use_precomputed_features = use_precomputed_features

        if use_precomputed_features:
            # Load precomputed features and paths
            self.state_features = np.load(os.path.join(data_dir, 'state_features.npy'))
            self.action_features = np.load(os.path.join(data_dir, 'action_features.npy'))
            with open(os.path.join(data_dir, 'states.json'), 'r') as f:
                self.states = json.load(f)
            with open(os.path.join(data_dir, 'actions.json'), 'r') as f:
                self.actions = json.load(f)
            with open(os.path.join(data_dir, 'screenshot_paths.json'), 'r') as f:
                self.screenshot_paths = json.load(f)
            self.length = len(self.states) - seq_len + 1
        else:
            # Old behavior
            with open(os.path.join(data_dir, 'states.json'), 'r') as f:
                self.states = json.load(f)
            with open(os.path.join(data_dir, 'actions.json'), 'r') as f:
                self.actions = json.load(f)
            self.screenshot_dir = os.path.join(data_dir, 'screenshots')
            self.length = len(self.states) - seq_len + 1

    def __len__(self):
        return max(0, self.length)

    def __getitem__(self, idx):
        if self.use_precomputed_features:
            # Use precomputed features and screenshot paths
            idxs = range(idx, idx + self.seq_len)
            images = []
            for i in idxs:
                img_path = self.screenshot_paths[i]
                try:
                    img = Image.open(img_path).convert('RGB')
                    img = self.transform(img)
                except Exception:
                    img = torch.zeros(3, 224, 224)
                images.append(img)
            images = torch.stack(images)
            state_tensor = torch.tensor(self.state_features[idx:idx+self.seq_len], dtype=torch.float32)
            action_tensor = torch.tensor(self.action_features[idx:idx+self.seq_len], dtype=torch.float32)
            return images, state_tensor, action_tensor
        else:
            # Old behavior
            states_seq = self.states[idx:idx+self.seq_len]
            actions_seq = self.actions[idx:idx+self.seq_len]
            images = []
            state_vecs = []
            action_vecs = []
            for j, (s, a) in enumerate(zip(states_seq, actions_seq)):
                # Image
                img_path = os.path.join(self.screenshot_dir, s.get('screenshot_filename', ''))
                try:
                    img = Image.open(img_path).convert('RGB')
                    img = self.transform(img)
                except Exception:
                    img = torch.zeros(3, 224, 224)  # fallback blank image
                images.append(img)
                # Feature extraction with temporal context
                prev_s = states_seq[j-1] if j > 0 else None
                prev_a = actions_seq[j-1] if j > 0 else None
                state_vecs.append(extract_state_features(s, self.n_npcs, self.n_objects, self.n_widgets, prev_state=prev_s))
                action_vecs.append(extract_action_features(a, prev_action=prev_a))
            images = torch.stack(images)  # (T, C, H, W)
            state_tensor = torch.tensor(np.stack(state_vecs), dtype=torch.float32)  # (T, F)
            action_tensor = torch.tensor(np.stack(action_vecs), dtype=torch.float32)  # (T, A)
            return images, state_tensor, action_tensor

# Example usage:
# dataset = ILSequenceDataset('path/to/data', seq_len=10, use_precomputed_features=True)
# loader = DataLoader(dataset, batch_size=8, shuffle=True)

# TODO: Add support for variable-length sequences, masking, and batching
# TODO: Implement proper state/action feature extraction 