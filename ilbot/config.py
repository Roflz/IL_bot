#!/usr/bin/env python3
"""
Configuration file for OSRS imitation learning
"""

# Canonical event class order used everywhere (indices 0..3)
EVENT_ORDER = ["CLICK", "KEY", "SCROLL", "MOVE"]

# Time (seconds). Targets are already scaled in v2; we clamp for metrics.
time_clip = 3.0
time_div = 1000
time_quantiles = [0.1, 0.5, 0.9]
# bias so softplus(bias) ≈ dataset median (~11ms for true Δt, ~24ms for prior)
# From your data: targets are ~29.3ms, so bias should be log(29.3/1000) ≈ -3.4
time_head_bias_default = -3.4

# XY heteroscedastic floor (pixels)
xy_min_sigma = 3.0

# Event head CE weights and loss weight (CLICK, KEY, SCROLL, MOVE)
# Up-weight rare events: CLICK and SCROLL
event_cls_weights = [8.0, 1.0, 12.0, 1.0]
event_loss_weight = 1.0

# Loss weights for different components
# Aggressive weights to force learning and prevent collapse
loss_weights = {
    "event": 5.0,    # Much stronger event weight to break collapse
    "time": 3.0,     # Stronger time weight to prevent all-zeros
    "xy": 2.0,       # Increase XY weight for better coordinate learning
    "btn": 1.0,      # Keep legacy weights
    "ka": 1.0,
    "kid": 1.0,
    "sy": 1.0
}

# Reporting configuration
report_k_top = 10
report_examples = 20
report_time_clamped_reference = False
debug_time = False
