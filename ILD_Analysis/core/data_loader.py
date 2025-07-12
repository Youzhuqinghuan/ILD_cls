"""
Data loader module for ILD Analysis
Provides unified interface for loading CT images and lesion predictions
"""

import os
import numpy as np
import SimpleITK as sitk
from typing import Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class ILDDataLoader:
    """Unified data loader for CT images and lesion predictions"""
    
    def __init__(self, ct_dir: str, lesion_dir: str):
        """
        Initialize data loader
        
        Args:
            ct_dir: Directory containing CT images
            lesion_dir: Directory containing lesion predictions
        """
        self.ct_dir = ct_dir
        self.lesion_dir = lesion_dir
        
    def load_case(self, case_name: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], 
                                                Optional[Tuple], Optional[sitk.Image]]:
        """
        Load CT image and lesion prediction for a case
        
        Args:
            case_name: Case identifier (e.g., "NSIP1")
            
        Returns:
            Tuple of (ct_array, lesion_array, spacing, ct_sitk_image)
            Returns None values if loading fails
        """
        try:
            # Construct file paths
            ct_filename = f"CT_{case_name}_0000.nii.gz"
            lesion_filename = f"CT_{case_name}.nii.gz"
            
            ct_path = os.path.join(self.ct_dir, ct_filename)
            lesion_path = os.path.join(self.lesion_dir, lesion_filename)
            
            # Check if files exist
            if not os.path.exists(ct_path):
                logger.error(f"CT file not found: {ct_path}")
                return None, None, None, None
                
            if not os.path.exists(lesion_path):
                logger.error(f"Lesion file not found: {lesion_path}")
                return None, None, None, None
            
            # Load CT image
            ct_sitk = sitk.ReadImage(ct_path)
            ct_array = sitk.GetArrayFromImage(ct_sitk)
            spacing = ct_sitk.GetSpacing()[::-1]  # Convert to (z, y, x) order
            
            # Load lesion prediction
            lesion_sitk = sitk.ReadImage(lesion_path)
            lesion_array = sitk.GetArrayFromImage(lesion_sitk)
            
            # Validate shapes match
            if ct_array.shape != lesion_array.shape:
                logger.error(f"Shape mismatch for {case_name}: "
                           f"CT {ct_array.shape} vs Lesion {lesion_array.shape}")
                return None, None, None, None
            
            logger.info(f"Loaded {case_name}: shape={ct_array.shape}, spacing={spacing}")
            
            return ct_array, lesion_array, spacing, ct_sitk
            
        except Exception as e:
            logger.error(f"Error loading case {case_name}: {str(e)}")
            return None, None, None, None
    
    def find_available_cases(self) -> list:
        """
        Find all cases that have both CT and lesion files
        
        Returns:
            List of case names that have both files available
        """
        available_cases = []
        
        # Get all lesion files (these define available cases)
        if not os.path.exists(self.lesion_dir):
            logger.warning(f"Lesion directory not found: {self.lesion_dir}")
            return available_cases
            
        lesion_files = [f for f in os.listdir(self.lesion_dir) 
                       if f.startswith("CT_") and f.endswith(".nii.gz")]
        
        for lesion_file in lesion_files:
            # Extract case name: CT_NSIP1.nii.gz -> NSIP1
            case_name = lesion_file.replace("CT_", "").replace(".nii.gz", "")
            
            # Check if corresponding CT file exists
            ct_filename = f"CT_{case_name}_0000.nii.gz"
            ct_path = os.path.join(self.ct_dir, ct_filename)
            
            if os.path.exists(ct_path):
                available_cases.append(case_name)
            else:
                logger.warning(f"Missing CT file for case {case_name}: {ct_filename}")
        
        logger.info(f"Found {len(available_cases)} available cases")
        return sorted(available_cases)
    
    def validate_case(self, case_name: str) -> Dict[str, Any]:
        """
        Validate a case and return metadata
        
        Args:
            case_name: Case identifier
            
        Returns:
            Dictionary with validation results and metadata
        """
        result = {
            "case_name": case_name,
            "valid": False,
            "ct_exists": False,
            "lesion_exists": False,
            "shape_match": False,
            "ct_shape": None,
            "lesion_shape": None,
            "spacing": None,
            "error": None
        }
        
        try:
            ct_filename = f"CT_{case_name}_0000.nii.gz"
            lesion_filename = f"CT_{case_name}.nii.gz"
            
            ct_path = os.path.join(self.ct_dir, ct_filename)
            lesion_path = os.path.join(self.lesion_dir, lesion_filename)
            
            # Check file existence
            result["ct_exists"] = os.path.exists(ct_path)
            result["lesion_exists"] = os.path.exists(lesion_path)
            
            if not result["ct_exists"] or not result["lesion_exists"]:
                result["error"] = "Missing files"
                return result
            
            # Check shapes and spacing
            ct_sitk = sitk.ReadImage(ct_path)
            lesion_sitk = sitk.ReadImage(lesion_path)
            
            ct_array = sitk.GetArrayFromImage(ct_sitk)
            lesion_array = sitk.GetArrayFromImage(lesion_sitk)
            
            result["ct_shape"] = ct_array.shape
            result["lesion_shape"] = lesion_array.shape
            result["spacing"] = ct_sitk.GetSpacing()[::-1]
            result["shape_match"] = ct_array.shape == lesion_array.shape
            
            if result["shape_match"]:
                result["valid"] = True
            else:
                result["error"] = "Shape mismatch"
                
        except Exception as e:
            result["error"] = str(e)
        
        return result