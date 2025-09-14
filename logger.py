"""
Simple logger module for WavePilot bridge.
"""

import logging
import os
import time
from multiprocessing import current_process

from constants import LOG_FOLDER
from logging.handlers import QueueHandler


def setup_logger(name, level=logging.INFO):
    """Set up a simple logger for the bridge executable."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # Config handlers if not already configured
    if not logger.handlers:
        # Use console handler for the bridge executable
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        logger.addHandler(console_handler)

    return logger
