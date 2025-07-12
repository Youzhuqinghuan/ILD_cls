"""
Configuration settings for ILD Analysis
"""

import os
from typing import Dict, Any

class Config:
    """Configuration class for ILD Analysis pipeline"""
    
    # Data paths
    CT_IMAGES_DIR = "dataset/images"
    PREDICTED_LABELS_DIR = "ILD_Segmentation/segs"
    OUTPUT_DIR = "ILD_Analysis/results"
    
    # File naming patterns
    CT_FILE_PATTERN = "CT_{case_name}_0000.nii.gz"
    LESION_FILE_PATTERN = "CT_{case_name}.nii.gz"
    
    # Segmentation parameters
    LUNG_THRESHOLD = -300  # HU threshold for lung segmentation
    
    # Feature extraction parameters
    UPPER_LOWER_PERCENTILE = 60  # Percentile for upper/lower lung division
    PERIPHERAL_THRESHOLD_MM = 10  # mm threshold for peripheral classification
    PERIPHERAL_RATIO_THRESHOLD = 0.20  # Relative threshold for peripheral
    AXIAL_CLASSIFICATION_THRESHOLD = 0.70  # Threshold for axial distribution
    SUBPLEURAL_THRESHOLD_MM = 3.0  # mm threshold for subpleural involvement
    
    # Lesion type labels (from ILD_Segmentation)
    LESION_LABELS = {
        0: "background",
        1: "honeycomb", 
        2: "reticulation",
        3: "ground_glass_opacity",  # GGO
        4: "consolidation"
    }
    
    # Pattern classification rules
    PATTERN_RULES = {
        "UIP": {
            "required_lesions": ["honeycomb"],
            "subpleural": True,
            "distribution": "lower_predominant"
        },
        "NSIP": {
            "excluded_lesions": ["honeycomb"],
            "allowed_lesions": ["ground_glass_opacity", "reticulation"],
            "distribution": ["diffuse", "upper_predominant"]
        },
        "OP": {
            "dominant_lesions": ["ground_glass_opacity", "consolidation"],
            "pattern": "migratory"
        }
    }
    
    @classmethod
    def get_ct_path(cls, case_name: str, base_dir: str = None) -> str:
        """Get CT image file path for a case"""
        base_dir = base_dir or cls.CT_IMAGES_DIR
        filename = cls.CT_FILE_PATTERN.format(case_name=case_name)
        return os.path.join(base_dir, filename)
    
    @classmethod
    def get_lesion_path(cls, case_name: str, base_dir: str = None) -> str:
        """Get lesion prediction file path for a case"""
        base_dir = base_dir or cls.PREDICTED_LABELS_DIR
        filename = cls.LESION_FILE_PATTERN.format(case_name=case_name)
        return os.path.join(base_dir, filename)
    
    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            attr: getattr(cls, attr) 
            for attr in dir(cls) 
            if not attr.startswith('_') and not callable(getattr(cls, attr))
        }