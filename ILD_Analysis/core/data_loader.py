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
    """Unified data loader for CT images, lesion predictions, and lung masks"""
    
    def __init__(self, ct_dir: str, lesion_dir: str, lung_mask_dir: str = None):
        """
        Initialize data loader
        
        Args:
            ct_dir: Directory containing CT images
            lesion_dir: Directory containing lesion predictions
            lung_mask_dir: Directory containing precomputed lung masks (optional)
        """
        self.ct_dir = ct_dir
        self.lesion_dir = lesion_dir
        self.lung_mask_dir = lung_mask_dir
        
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
    
    def load_lung_masks(self, case_name: str) -> Dict[str, Optional[np.ndarray]]:
        """
        Load precomputed lung masks for a case
        
        Args:
            case_name: Case identifier (e.g., "NSIP1")
            
        Returns:
            Dictionary with lung mask arrays (lung_mask, upper_lung, lower_lung, lobe_mask)
            Returns None values if files don't exist or loading fails
        """
        masks = {
            'lung_mask': None,
            'upper_lung': None,
            'lower_lung': None,
            'lobe_mask': None
        }
        
        if not self.lung_mask_dir or not os.path.exists(self.lung_mask_dir):
            logger.debug(f"Lung mask directory not available: {self.lung_mask_dir}")
            return masks
        
        try:
            # Define expected lung mask files
            mask_files = {
                'lung_mask': f"CT_{case_name}_lung_mask.nii.gz",
                'upper_lung': f"CT_{case_name}_upper_lung.nii.gz", 
                'lower_lung': f"CT_{case_name}_lower_lung.nii.gz",
                'lobe_mask': f"CT_{case_name}_lobe_mask.nii.gz"  # Optional
            }
            
            # Load each mask file
            for mask_type, filename in mask_files.items():
                mask_path = os.path.join(self.lung_mask_dir, filename)
                
                if os.path.exists(mask_path):
                    try:
                        mask_sitk = sitk.ReadImage(mask_path)
                        mask_array = sitk.GetArrayFromImage(mask_sitk)
                        masks[mask_type] = mask_array
                        logger.debug(f"Loaded {mask_type} for {case_name}: shape={mask_array.shape}")
                    except Exception as e:
                        logger.warning(f"Failed to load {mask_type} for {case_name}: {str(e)}")
                        masks[mask_type] = None
                else:
                    # Lobe mask is optional, others are expected
                    if mask_type != 'lobe_mask':
                        logger.debug(f"{mask_type} file not found for {case_name}: {filename}")
            
            # Check if we have at least the basic lung mask
            if masks['lung_mask'] is not None:
                logger.info(f"Loaded precomputed lung masks for {case_name}")
            else:
                logger.debug(f"No precomputed lung masks available for {case_name}")
                
        except Exception as e:
            logger.error(f"Error loading lung masks for {case_name}: {str(e)}")
        
        return masks
    
    def load_case_with_lung_masks(self, case_name: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], 
                                                               Optional[Tuple], Optional[sitk.Image], Dict[str, Optional[np.ndarray]]]:
        """
        Load CT image, lesion prediction, and precomputed lung masks for a case
        
        Args:
            case_name: Case identifier (e.g., "NSIP1")
            
        Returns:
            Tuple of (ct_array, lesion_array, spacing, ct_sitk_image, lung_masks_dict)
        """
        # Load basic case data
        ct_array, lesion_array, spacing, ct_sitk = self.load_case(case_name)
        
        # Load lung masks
        lung_masks = self.load_lung_masks(case_name)
        
        return ct_array, lesion_array, spacing, ct_sitk, lung_masks
    
    def check_lung_masks_available(self, case_name: str) -> Dict[str, bool]:
        """
        Check which precomputed lung masks are available for a case
        
        Args:
            case_name: Case identifier
            
        Returns:
            Dictionary indicating availability of each mask type
        """
        availability = {
            'lung_mask': False,
            'upper_lung': False,
            'lower_lung': False,
            'lobe_mask': False
        }
        
        if not self.lung_mask_dir or not os.path.exists(self.lung_mask_dir):
            return availability
        
        mask_files = {
            'lung_mask': f"CT_{case_name}_lung_mask.nii.gz",
            'upper_lung': f"CT_{case_name}_upper_lung.nii.gz",
            'lower_lung': f"CT_{case_name}_lower_lung.nii.gz",
            'lobe_mask': f"CT_{case_name}_lobe_mask.nii.gz"
        }
        
        for mask_type, filename in mask_files.items():
            mask_path = os.path.join(self.lung_mask_dir, filename)
            availability[mask_type] = os.path.exists(mask_path)
        
        return availability
    
    def find_cases_with_lung_masks(self) -> list:
        """
        Find all cases that have precomputed lung masks available
        
        Returns:
            List of case names with lung masks
        """
        if not self.lung_mask_dir or not os.path.exists(self.lung_mask_dir):
            return []
        
        cases_with_masks = []
        
        # Look for lung mask files
        mask_files = [f for f in os.listdir(self.lung_mask_dir) 
                     if f.startswith("CT_") and f.endswith("_lung_mask.nii.gz")]
        
        for mask_file in mask_files:
            # Extract case name: CT_NSIP1_lung_mask.nii.gz -> NSIP1
            case_name = mask_file.replace("CT_", "").replace("_lung_mask.nii.gz", "")
            cases_with_masks.append(case_name)
        
        return sorted(cases_with_masks)
    
    def validate_case_with_lung_masks(self, case_name: str) -> Dict[str, Any]:
        """
        Validate a case including lung mask availability
        
        Args:
            case_name: Case identifier
            
        Returns:
            Dictionary with validation results including lung mask status
        """
        # Get basic validation
        result = self.validate_case(case_name)
        
        # Add lung mask information
        result["lung_masks"] = self.check_lung_masks_available(case_name)
        result["has_precomputed_masks"] = any(result["lung_masks"].values())
        
        return result