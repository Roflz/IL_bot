#!/usr/bin/env python3
"""Logging setup for the Bot Controller GUI"""

import logging
import sys
import os

TRACE_LEVEL_NUM = 9  # between NOTSET(0) and DEBUG(10)
if not hasattr(logging, "TRACE"):
    logging.addLevelName(TRACE_LEVEL_NUM, "TRACE")
    def trace(self, msg, *args, **kwargs):
        if self.isEnabledFor(TRACE_LEVEL_NUM):
            self._log(TRACE_LEVEL_NUM, msg, args, **kwargs)
    logging.Logger.trace = trace


def init_logging(level=logging.DEBUG):
    """Initialize global logging configuration"""
    # Env switch: BOTGUI_TRACE=1 -> TRACE level globally
    if os.environ.get("BOTGUI_TRACE") == "1":
        level = TRACE_LEVEL_NUM
    logging.basicConfig(
        level=level, 
        stream=sys.stdout,
        format="%(asctime)s [%(levelname)s] %(name)s %(threadName)s %(filename)s:%(lineno)d: %(message)s"
    )
    # Make sure child loggers aren't filtered off
    logging.getLogger().setLevel(level)
