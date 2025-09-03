# ilbot/data/contracts.py
from typing import Dict, Tuple
import torch

# Screen bounds (pixels) used throughout for XY
SCREEN_W = 1920.0
SCREEN_H = 1080.0

# Canonical event mapping (SINGLE SOURCE OF TRUTH)
EVENT_ID_TO_NAME = {0: "CLICK", 1: "KEY", 2: "SCROLL", 3: "MOVE"}
EVENT_NAME_TO_ID = {v:k for k,v in EVENT_ID_TO_NAME.items()}

def derive_event_targets_from_marks(targets: torch.Tensor) -> torch.Tensor:
    """
    targets: [B,A,7] columns: [0] time_ms, [1] x_px, [2] y_px,
                               [3] button, [4] key_action, [5] key_id, [6] scroll_y
    return:  [B,A] event ids with default MOVE=3
    """
    assert targets.dim() == 3 and targets.size(-1) == 7, "targets must be [B,A,7]"
    device = targets.device
    B, A, _ = targets.shape
    ev = torch.full((B, A), 3, dtype=torch.long, device=device)  # MOVE default
    button = targets[..., 3].long()
    key_action = targets[..., 4].long()
    scroll_y = targets[..., 6].long()
    ev = torch.where(button != 0, torch.zeros_like(ev), ev)          # CLICK=0
    ev = torch.where(key_action != 0, torch.ones_like(ev), ev)       # KEY=1
    ev = torch.where(scroll_y != 0, torch.full_like(ev, 2), ev)       # SCROLL=2
    return ev

def build_valid_mask(targets: torch.Tensor) -> torch.Tensor:
    """
    A slot is valid if there is a real action:
    define validity as (time_ms > 0) OR any mark present.
    Returns BoolTensor [B,A]
    """
    time_ms = targets[..., 0]
    marks = torch.stack([
        (targets[..., 3] != 0),   # button (0 = inactive)
        (targets[..., 4] != 0),   # key_action (0 = inactive)
        (targets[..., 5] != 0),   # key_id (0 = inactive)
        (targets[..., 6] != 0),   # scroll_y (0 = inactive)
    ], dim=-1)
    any_mark = marks.any(dim=-1)
    valid = (time_ms > 0) | any_mark
    return valid.bool()

def assert_batch_contract(batch: Dict, *, expect_temporal_window: int = None) -> None:
    """
    Enforce the single batch format contract.
    batch keys:
      temporal_sequence: Float [B,T,Dg]
      action_sequence:   Float [B,T,A,Fa]
      targets:           Float [B,A,7]    (time_ms, x, y, button, key_action, key_id, scroll_y)
      valid_mask:        Bool  [B,A]
    Also sanity-check value ranges and %valid.
    """
    req = ["temporal_sequence","action_sequence","targets","valid_mask"]
    for k in req:
        assert k in batch, f"Missing batch key: {k}"

    ts = batch["temporal_sequence"]
    ac = batch["action_sequence"]
    tg = batch["targets"]
    vm = batch["valid_mask"]

    assert ts.dtype.is_floating_point and ac.dtype.is_floating_point and tg.dtype.is_floating_point, "tensors must be float"
    assert vm.dtype == torch.bool, "valid_mask must be boolean"

    assert ts.dim()==3 and ac.dim()==4 and tg.dim()==3 and vm.dim()==2, "rank mismatch"
    B = ts.size(0)
    assert ac.size(0)==B and tg.size(0)==B and vm.size(0)==B, "batch dim mismatch"

    T = ts.size(1)
    if expect_temporal_window is not None:
        assert T == expect_temporal_window, f"expected temporal window {expect_temporal_window}, got {T}"

    A = ac.size(2)
    assert tg.size(1)==A and vm.size(1)==A, "action slots mismatch"

    assert tg.size(-1)==7, "targets last dim must be 7"
    # sanity on %valid
    frac_valid = vm.float().mean().item()
    assert 0.05 <= frac_valid <= 0.70, f"valid_mask density seems off: {frac_valid:.3f} (expected ~0.10–0.30)"
