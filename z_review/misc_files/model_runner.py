# model_runner.py
import os
import numpy as np
from pathlib import Path

try:
    import torch
    import torch.nn as nn
except Exception:
    torch, nn = None, None

# Centralized path resolution
BASE_DIR = Path(__file__).resolve().parent

def rp(*parts: str) -> Path:
    """Repo-local absolute path builder."""
    return (BASE_DIR / Path(*parts)).resolve()

DATA_DIR = rp("data")
FEATURES_DIR = rp("data", "features")
MODEL_PATH = rp("training_results", "model_weights.pth")

if nn is not None:
    class _StepMLP(nn.Module):
        """Tiny per-timestep MLP: [features(128) + actions(8)] -> 8."""
        def __init__(self, num_features=128, num_actions=8, hidden=128):
            super().__init__()
            self.net = nn.Sequential(
                nn.LazyLinear(hidden),
                nn.ReLU(),
                nn.Linear(hidden, num_actions),
            )
        def forward(self, temporal, actionin):
            # Handle both 3D and 4D actionin tensors
            if actionin.ndim == 4:
                # Flatten (B, T, 101, 8) to (B, T, 101*8) for the fallback
                B, T, H, W = actionin.shape
                actionin = actionin.reshape(B, T, H * W)
            
            x = torch.cat([temporal, actionin], dim=-1)  # (B, T, 128+num_actions)
            return self.net(x)  # (B, T, num_actions)

class ModelRunner:
    _singleton = None
    
    # Model constants
    seq_len = 10            # T
    num_features = 128
    max_actions = 100       # micro-actions per step
    action_vec = 8          # fields per action
    model_action_rows = 101 # 1 header + 100 actions
    
    def __init__(self, model_path: str = None, device: str | None = None):
        if model_path is None:
            model_path = MODEL_PATH
        # Default attributes
        self.num_actions = 8        # size of action vector (one-hot)
        
        if torch is None or nn is None:
            self.torch_ok = False
            self.model = None
            self.device = "cpu"
            self.device_str = "cpu"
            return
            
        self.torch_ok = True
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device_str = "cuda" if (torch and torch.cuda.is_available()) else "cpu"
        
        # Try to load real model first
        self.model = self._try_load_real_model(model_path)
        
        # Fallback to tiny MLP if loading fails
        if self.model is None:
            print("ModelRunner: falling back to tiny MLP")
            self.model = _StepMLP(num_features=128, num_actions=self.num_actions).to(self.device)
        
        try:
            self.model.device = torch.device(self.device)
        except Exception:
            pass
        
        self.model.eval()

    def _try_load_real_model(self, model_path):
        """Try to load the real model and infer num_actions and seq_len."""
        if not model_path or not os.path.isfile(model_path):
            print(f"ModelRunner: weights not found at {model_path}")
            return None
            
        try:
            state = torch.load(model_path, map_location=self.device)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            
            # Try to import and construct the real model
            try:
                from model.imitation_hybrid_model import ImitationHybridModel
                # Infer num_actions from state dict
                for key in state.keys():
                    if key.endswith('.weight') and 'action' in key.lower():
                        if 'out' in key.lower() or 'final' in key.lower():
                            # This is likely the output layer
                            if state[key].shape[0] > 0:
                                self.num_actions = state[key].shape[0]
                                break
                
                # Create model with inferred parameters
                model = ImitationHybridModel(num_features=128, num_actions=self.num_actions)
                model.load_state_dict(state, strict=False)  # tolerate mismatches
                model = model.to(self.device)
                
                print(f"ModelRunner: loaded real model with {self.num_actions} actions")
                return model
                
            except ImportError:
                print("ModelRunner: ImitationHybridModel not available, using fallback")
                return None
                
        except Exception as e:
            print(f"ModelRunner: WARNING could not load weights: {e}")
            return None

    @classmethod
    def instance(cls):
        if cls._singleton is None:
            cls._singleton = cls()
        return cls._singleton

    def predict(self, temporal, actionin):
        """
        temporal: (B, T, 128) float32
        actionin: (B, T, 101, 8) float32   # header + micro-actions
        return:  (B, 101, 8)               # next-600ms predicted actions
        """
        if torch is None or not self.torch_ok:
            if isinstance(temporal, np.ndarray):
                B, T_in = temporal.shape[0], temporal.shape[1]
            else:
                B, T_in = 1, self.seq_len
            return np.zeros((B, 101, 8), dtype=np.float32)
        
        with torch.no_grad():
            t = temporal if torch.is_tensor(temporal) else torch.as_tensor(temporal)
            a = actionin if torch.is_tensor(actionin) else torch.as_tensor(actionin)
            
            t = t.to(self.device, dtype=torch.float32, non_blocking=True)
            a = a.to(self.device, dtype=torch.float32, non_blocking=True)
            
            # Get input dimensions
            T_in = int(t.shape[1]) if t.ndim >= 2 else 1
            T_model = self.seq_len
            A_feat = self.num_features # 128
            A_action_vec = self.action_vec # 8
            A_model_action_rows = self.model_action_rows # 101
            
            # Ensure correct shapes for temporal (B, T, 128)
            if t.ndim != 3:
                t = t.reshape(1, -1, A_feat)
            B = t.shape[0]
            
            # Ensure correct shapes for actionin (B, T, 101, 8)
            if a.ndim != 4:
                if a.ndim == 3:
                    # If actionin is (B, T, A) where A != 101*8, reshape it
                    if a.shape[-1] != A_model_action_rows * A_action_vec:
                        # Reshape to (B, T, 101, 8) by padding/truncating
                        a = a.reshape(B, T_in, -1)
                        if a.shape[-1] < A_model_action_rows * A_action_vec:
                            # Pad with zeros
                            pad_size = A_model_action_rows * A_action_vec - a.shape[-1]
                            a = torch.nn.functional.pad(a, (0, pad_size))
                        else:
                            # Truncate
                            a = a[..., :A_model_action_rows * A_action_vec]
                        a = a.reshape(B, T_in, A_model_action_rows, A_action_vec)
                    else:
                        # Reshape (B, T, 101*8) to (B, T, 101, 8)
                        a = a.reshape(B, T_in, A_model_action_rows, A_action_vec)
                else:
                    # Create default action tensor
                    a = torch.zeros((B, T_in, A_model_action_rows, A_action_vec), 
                                  dtype=torch.float32, device=self.device)
            
            # Pad or truncate to model's expected sequence length
            if T_in != T_model:
                if T_in < T_model:
                    pad_size = T_model - T_in
                    t = torch.nn.functional.pad(t, (0, 0, 0, pad_size))
                    a = torch.nn.functional.pad(a, (0, 0, 0, 0, 0, pad_size))
                else:
                    t = t[:, :T_model, :]
                    a = a[:, :T_model, :, :]
            
            # Run inference
            y = self.model(t, a)  # This should return (B, 101, 8) or (B, T, num_actions) for fallback
            
            # Handle different output shapes
            if y.shape[-2:] == (A_model_action_rows, A_action_vec):
                # Real model output: (B, 101, 8) - no reshaping needed
                pass
            elif y.shape[-1] == self.num_actions:
                # Fallback MLP output: (B, T, num_actions) - reshape to expected format
                # For the fallback, we'll create a simple output structure
                B_out, T_out = y.shape[:2]
                y_reshaped = torch.zeros((B_out, A_model_action_rows, A_action_vec), 
                                       dtype=y.dtype, device=y.device)
                
                # Put the MLP output in the first row as a simple action
                if T_out > 0:
                    # Take the last timestep and put it in the first action row
                    last_output = y[:, -1, :]  # (B, num_actions)
                    # Pad or truncate to fit in the first action row
                    if last_output.shape[-1] <= A_action_vec:
                        y_reshaped[:, 1, :last_output.shape[-1]] = last_output
                    else:
                        y_reshaped[:, 1, :] = last_output[:, :A_action_vec]
                    
                    # Set count to 1 in header row
                    y_reshaped[:, 0, 0] = 1.0
                
                y = y_reshaped
            else:
                # Unexpected output shape - create default
                y = torch.zeros((B, A_model_action_rows, A_action_vec), 
                              dtype=torch.float32, device=self.device)
            
            # Convert to numpy and return
            return y.cpu().numpy()
