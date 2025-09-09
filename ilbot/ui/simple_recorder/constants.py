# constants.py
from pathlib import Path

import pyautogui

# keep identical defaults/behavior
pyautogui.FAILSAFE = True  # moving mouse to a corner aborts

AUTO_REFRESH_MS      = 100
AUTO_RUN_TICK_MS     = 250    # scheduler tick for the UI/auto loop
PRE_ACTION_DELAY_MS  = 250    # delay before performing any action (click/key)
RULE_WAIT_TIMEOUT_MS = 10_000 # pre/post condition wait timeout

SESSIONS_DIR = Path(r"D:\repos\bot_runelite_IL\data\recording_sessions")

# Grand Exchange region bounds (hardcoded)
GE_MIN_X = 3155
GE_MAX_X = 3173
GE_MIN_Y = 3479
GE_MAX_Y = 3498

# Edgeville Bank bounding box (world coordinates)
EDGE_BANK_MIN_X = 3092
EDGE_BANK_MAX_X = 3098
EDGE_BANK_MIN_Y = 3488
EDGE_BANK_MAX_Y = 3498