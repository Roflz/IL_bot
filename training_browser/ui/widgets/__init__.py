"""
Widgets Package

Reusable UI widgets for the training browser.
"""

from .scrollable_frame import ScrollableFrame
from .feature_table import FeatureTableView
from .target_view import TargetView
from .action_tensors_view import ActionTensorsView
from .sequence_alignment_view import SequenceAlignmentView
from .feature_analysis_view import FeatureAnalysisView
from .normalization_view import NormalizationView

__all__ = [
    "ScrollableFrame",
    "FeatureTableView",
    "TargetView",
    "ActionTensorsView",
    "SequenceAlignmentView",
    "FeatureAnalysisView",
    "NormalizationView"
]
