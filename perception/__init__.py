"""
Perception Module

Provides perception capabilities for robot manipulation:
- SAM3 segmentation
- Position calculation from cameras
"""

from .sam3_segmenter import SAM3Segmenter, create_segmenter
from .position_calculator import (
    HeadCameraCalculator,
    HandEyeCalculator,
    create_head_calculator,
    create_handeye_calculator,
    estimate_orientation,
)

__all__ = [
    # SAM3
    'SAM3Segmenter',
    'create_segmenter',
    # Position calculators
    'HeadCameraCalculator',
    'HandEyeCalculator',
    'create_head_calculator',
    'create_handeye_calculator',
    'estimate_orientation',
]