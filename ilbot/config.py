#!/usr/bin/env python3
"""
Configuration file for OSRS imitation learning
"""

# Event classification reweighting (MOVE, CLICK, KEY, SCROLL)
# Rationale: dataset is ~95% MOVE; these tempered weights keep rare classes learning
# without exploding gradients. You can set to None to auto-compute from label counts.
event_cls_weights = [1.0, 8.0, 6.0, 12.0]  # or None → auto (tempered inverse-frequency)

# Enforce physically non-negative time at the model head.
# If True, time head uses softplus; negatives cannot occur.
# If False, reporting will still print negative stats un-clamped.
time_positive = False

# Validation report knobs
report_examples = 8            # how many raw example predictions to print
report_k_top = 5               # top-k for key_id and scroll_y in report
exclusive_event = True         # derive event by argmax(event_logits); forbids MULTI in reports

# Optional: also show a reference "clamped-to-0" mean for time (raw stays raw)
report_time_clamped_reference = False
