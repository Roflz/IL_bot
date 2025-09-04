# ilbot/config.py
from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class Config:
    data_dir: str
    batch_size: int = 64
    num_workers: int = 0  # set >0 later if desired
    seed: int = 1337
    # Training-relevant constants (used later, defined now for clarity)
    max_actions: int = 100
    horizon_s: float = 0.6
    # Optional enum sizes (dataset may infer; keep here for visibility)
    enum_sizes: Optional[Dict[str, int]] = None  # {'button':3,'key_action':3,'key_id':505,'scroll':3}
    # Training hyperparameters
    epochs: int = 10
    lr: float = 2.5e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    amp: bool = True
    ckpt_dir: str = "checkpoints"
    run_name: Optional[str] = None  # default: auto timestamp
    patience: int = 5               # early stopping on val loss
    log_interval: int = 50          # steps

    # ---- New: imbalance & loss shaping (defaults keep old behavior)
    event_weighting: str = "none"   # {"none","inverse","inv_sqrt"}
    focal_gamma: float = 0.0        # 0.0 disables focal loss (pure CE)
    xy_weight: float = 0.1          # was fixed in loss; now configurable
    # Decoder XY logσ clamp; lets you widen uncertainty range if desired
    logsig_min: float = -3.0
    logsig_max: float = 0.0
