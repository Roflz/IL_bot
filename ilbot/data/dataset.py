# ilbot/data/dataset.py
from typing import Dict, Tuple
from pathlib import Path
import json
import math
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from ilbot.config import Config
from ilbot.data.contracts import build_valid_mask

class NPYDataset(Dataset):
    """
    Loads three NPY files:
      gamestate_sequences.npy      [N, T, Dg]   float32
      action_input_sequences.npy   [N, T, A, Fa] float32
      action_targets.npy           [N, A, 7]    float32 (time_ms, x, y, button, key_action, key_id, scroll_y)
    Returns per-item dict (will be stacked by DataLoader):
      temporal_sequence [T,Dg], action_sequence [T,A,Fa], targets [A,7], valid_mask [A]
    """
    def __init__(self, cfg: Config, split: str):
        self.cfg = cfg
        self.split = split
        p = Path(cfg.data_dir).resolve()
        self.gs = np.load(p / "gamestate_sequences.npy").astype(np.float32)        # [N,T,Dg]
        self.ai = np.load(p / "action_input_sequences.npy").astype(np.float32)     # [N,T,A,Fa]
        self.tg = np.load(p / "action_targets.npy").astype(np.float32)             # [N,A,7]

        # ---- Explicit NaN/Inf guard: fail fast on bad source arrays
        for name, arr in (("gamestate_sequences", self.gs), ("action_input_sequences", self.ai), ("action_targets", self.tg)):
            if not np.isfinite(arr).all():
                bad = int(np.size(arr) - np.isfinite(arr).sum())
                raise ValueError(f"{name} contains {bad} non-finite values; fix your NPYs before training.")

        # Basic shape checks
        assert self.gs.ndim==3 and self.ai.ndim==4 and self.tg.ndim==3, "NPY rank mismatch"
        N = self.gs.shape[0]
        assert self.ai.shape[0]==N and self.tg.shape[0]==N, "NPY count mismatch"

        # --- Strict normalization: MUST read bounds from a single source of truth.
        # Anchor path relative to the data root to avoid CWD surprises:
        # <repo>/data/recording_sessions/.../<run>/06_final_training_data  ->  <repo>/data/data_profile/norm.json
        data_root = p.parents[2]   # .../<repo>/data
        if data_root.name != "data":
            raise RuntimeError(f"Expected data root to be .../data, got {data_root}")
        norm_path = (data_root / "data_profile" / "norm.json")
        if not norm_path.exists():
            raise FileNotFoundError(
                f"Missing normalization file: {norm_path}. "
                "Run `python scripts/data_audit.py` to generate it, then rerun."
            )
        try:
            norm = json.loads(norm_path.read_text())
            XMAX = float(norm["x_max"])
            YMAX = float(norm["y_max"])
            if not (XMAX > 0 and YMAX > 0 and math.isfinite(XMAX) and math.isfinite(YMAX)):
                raise ValueError("x_max/y_max must be positive finite numbers in norm.json")
        except Exception as e:
            raise RuntimeError(f"Failed to read/parse {norm_path}: {e}")

        # Verify raw data does not exceed the declared bounds (no silent clipping)
        raw_x_max = float(np.nanmax(self.tg[...,1])) if self.tg.size else 0.0
        raw_y_max = float(np.nanmax(self.tg[...,2])) if self.tg.size else 0.0
        eps = 1e-6
        if raw_x_max > XMAX + eps or raw_y_max > YMAX + eps:
            raise ValueError(
                f"Raw XY exceeds normalization bounds: "
                f"raw_x_max={raw_x_max:.6f} > x_max={XMAX:.6f} or "
                f"raw_y_max={raw_y_max:.6f} > y_max={YMAX:.6f}. "
                "Re-run the audit to update norm.json or correct the dataset."
            )

        # One-time dataset-wide normalized range check (no per-sample numpy in __getitem__)
        time_s  = self.tg[...,0] / 1000.0
        x_norm  = self.tg[...,1] / XMAX
        y_norm  = self.tg[...,2] / YMAX
        if np.nanmin(x_norm) < -eps or np.nanmax(x_norm) > 1 + eps:
            lo, hi = float(np.nanmin(x_norm)), float(np.nanmax(x_norm))
            raise ValueError(f"x_norm out of [0,1] dataset-wide: min={lo:.6f}, max={hi:.6f}")
        if np.nanmin(y_norm) < -eps or np.nanmax(y_norm) > 1 + eps:
            lo, hi = float(np.nanmin(y_norm)), float(np.nanmax(y_norm))
            raise ValueError(f"y_norm out of [0,1] dataset-wide: min={lo:.6f}, max={hi:.6f}")
        if np.nanmin(time_s) < -eps or np.nanmax(time_s) > 10 + eps:
            lo, hi = float(np.nanmin(time_s)), float(np.nanmax(time_s))
            raise ValueError(f"time_seconds out of [0,10] dataset-wide: min={lo:.6f}, max={hi:.6f}")

        self.N = N
        self.T = int(self.gs.shape[1])
        self.Dg = int(self.gs.shape[2])
        self.A = int(self.ai.shape[2])
        self.Fa = int(self.ai.shape[3])
        self.XMAX = XMAX
        self.YMAX = YMAX
        self.eps = eps

    def __len__(self):
        return self.N

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ts = torch.from_numpy(self.gs[idx])             # [T,Dg]
        ac = torch.from_numpy(self.ai[idx])             # [T,A,Fa]
        tg = torch.from_numpy(self.tg[idx])             # [A,7]
        
        # Normalize here for model: time ms -> seconds; pixels -> [0,1] using norm.json
        tg_normalized = tg.clone()
        tg_normalized[..., 0] = tg[..., 0] / 1000.0  # ms -> seconds
        tg_normalized[..., 1] = tg[..., 1] / self.XMAX
        tg_normalized[..., 2] = tg[..., 2] / self.YMAX

        # No per-item min/max checks here (enforced once in __init__)
        
        # Build a per-sample valid_mask [A]
        vm = build_valid_mask(tg.unsqueeze(0)).squeeze(0)  # [A]
        return {
            "temporal_sequence": ts,
            "action_sequence": ac,
            "targets": tg_normalized,
            "valid_mask": vm,
        }

def _seed_everything(seed: int):
    import random, numpy as np, torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def make_loaders(cfg: Config) -> Tuple[DataLoader, DataLoader, Dict]:
    """
    Returns train_loader, val_loader, and a small data_config dict derived from shapes.
    """
    _seed_everything(cfg.seed)
    ds = NPYDataset(cfg, "train")
    N = len(ds)
    n_val = max(1, int(0.2 * N))
    n_train = N - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(cfg.seed))

    def _collate(batch_list):
        # Stack each field
        keys = batch_list[0].keys()
        out = {}
        for k in keys:
            out[k] = torch.stack([b[k] for b in batch_list], dim=0)
        return out

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
                              collate_fn=_collate)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=True, drop_last=False,
                            collate_fn=_collate)

    data_config = {
        "gamestate_dim": ds.Dg,
        "temporal_window": ds.T,
        "max_actions": ds.A,        # use real A; we can clamp to cfg.max_actions later if needed
        "action_features": ds.Fa,
        # You can pass enum sizes via cfg.enum_sizes later; data doesn't encode them here.
        "event_types": 4,
    }
    return train_loader, val_loader, data_config
