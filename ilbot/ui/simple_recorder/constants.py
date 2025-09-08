# constants.py
import pyautogui

# keep identical defaults/behavior
pyautogui.FAILSAFE = True  # moving mouse to a corner aborts

AUTO_REFRESH_MS      = 100
AUTO_RUN_TICK_MS     = 250    # scheduler tick for the UI/auto loop
PRE_ACTION_DELAY_MS  = 250    # delay before performing any action (click/key)
RULE_WAIT_TIMEOUT_MS = 10_000 # pre/post condition wait timeout
