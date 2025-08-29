#!/usr/bin/env python3
"""State management for Bot Controller GUI"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any
from collections import deque
import numpy as np


@dataclass
class UIState:
    """UI state variables"""
    data_root: Path = Path("data")
    bot_mode: str = "bot1"
    live_mode: bool = False
    show_translations: bool = True
    feature_group_filter: str = "All"
    search_text: str = ""


@dataclass
class RuntimeState:
    """Runtime state for the application"""
    # Feature pipeline state
    features_buffer: deque = field(default_factory=lambda: deque(maxlen=10))
    actions_buffer: deque = field(default_factory=lambda: deque(maxlen=10))

    # Model state
    model_loaded: bool = False
    model_path: Optional[Path] = None

    # Live source state
    live_source_active: bool = False
    live_mode: bool = False
    last_update_time: Optional[float] = None

    # Prediction state
    predictions_enabled: bool = True
    track_user_input: bool = False

    # Status
    status_message: str = "Ready"
    error_message: Optional[str] = None


@dataclass
class FeatureData:
    """Feature data structure"""
    features: np.ndarray  # Shape: (10, 128)
    timestamp: float
    source: str


@dataclass
class ActionData:
    """Action data structure"""
    actions: list  # List of flattened action frames
    timestamp: float
    source: str


@dataclass
class PredictionData:
    """Prediction data structure"""
    action_frame: np.ndarray  # Flattened 600ms action frame
    timestamp: float
    confidence: Optional[float] = None


@dataclass
class BufferStatus:
    """Buffer status information"""
    features_count: int = 0
    actions_count: int = 0
    is_warm: bool = False
    last_update: Optional[float] = None
    source_mode: str = "unknown"
