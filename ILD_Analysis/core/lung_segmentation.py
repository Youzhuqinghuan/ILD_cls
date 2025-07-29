"""
Lung segmentation module for ILD Analysis
Uses lungmask for high-accuracy lung segmentation with fallback to legacy method
"""

import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# Try to import lungmask-based segmentator, fallback to legacy
try:
    from .lungmask_segmentation import LungmaskSegmentator as MainSegmentator
    LUNGMASK_AVAILABLE = True
    logger.info("Using lungmask-based segmentation as primary method")
except ImportError:
    from .lung_segmentation_legacy import LungSegmentator as MainSegmentator
    LUNGMASK_AVAILABLE = False
    logger.warning("lungmask not available, using legacy morphological segmentation")

# Alias for backward compatibility
LungSegmentator = MainSegmentator