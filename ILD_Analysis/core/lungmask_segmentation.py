"""
Lungmask-based lung segmentation module for ILD Analysis
Provides high-accuracy lung segmentation using pre-trained U-net models
"""

import numpy as np
import SimpleITK as sitk
from typing import Tuple, Optional
import logging
import os

logger = logging.getLogger(__name__)

# Optional import with fallback
try:
    from lungmask import LMInferer
    LUNGMASK_AVAILABLE = True
except ImportError:
    LUNGMASK_AVAILABLE = False
    logger.warning("lungmask not available. Install with: pip install lungmask")

class LungmaskSegmentator:
    """Lungmask-based lung segmentation with upper/lower division"""
    
    def __init__(self, model_name: str = 'R231', enable_lobe_segmentation: bool = False):
        """
        Initialize lungmask segmentator
        
        Args:
            model_name: Model to use ('R231' for basic lung segmentation)
            enable_lobe_segmentation: Whether to enable lobe-based segmentation (deprecated, always False)
        """
        self.model_name = model_name
        self.enable_lobe_segmentation = False  # Force disable lobe segmentation
        
        if not LUNGMASK_AVAILABLE:
            raise ImportError("lungmask package not available. Install with: pip install lungmask")
        
        # Initialize basic model only
        try:
            self.basic_inferer = LMInferer(modelname='R231')
            self.lobe_inferer = None  # No lobe segmentation
                
            logger.info(f"Initialized lungmask segmentator (model: {model_name}, percentile-based upper/lower division)")
            
        except Exception as e:
            logger.error(f"Failed to initialize lungmask models: {str(e)}")
            raise
    
    def segment_volume(self, ct_array: np.ndarray, spacing: Tuple[float, float, float]) -> np.ndarray:
        """
        Segment lung regions in a 3D CT volume using lungmask
        
        Args:
            ct_array: 3D CT array (Z, Y, X)
            spacing: Pixel spacing (z, y, x)
            
        Returns:
            3D binary mask of lung regions
        """
        try:
            # Convert to SimpleITK format
            sitk_image = sitk.GetImageFromArray(ct_array)
            sitk_image.SetSpacing(spacing[::-1])  # ITK uses (x, y, z) order
            
            # Apply lungmask segmentation
            logger.debug("Applying lungmask segmentation...")
            lung_mask = self.basic_inferer.apply(sitk_image)
            
            # lungmask.apply() already returns numpy array
            lung_mask = (lung_mask > 0).astype(np.uint8)
            
            logger.info(f"Segmented lung volume: {np.sum(lung_mask)} voxels")
            return lung_mask
            
        except Exception as e:
            logger.error(f"Error in lungmask segmentation: {str(e)}")
            return np.zeros_like(ct_array, dtype=np.uint8)
    
    def segment_with_lobes(self, ct_array: np.ndarray, spacing: Tuple[float, float, float]) -> dict:
        """
        Segment lung regions using basic segmentation and percentile-based upper/lower division
        
        Args:
            ct_array: 3D CT array (Z, Y, X)
            spacing: Pixel spacing (z, y, x)
            
        Returns:
            Dictionary with lung masks (no lobe information)
        """
        try:
            # Perform basic lung segmentation
            lung_mask = self.segment_volume(ct_array, spacing)
            
            if not np.any(lung_mask):
                logger.warning("No lung regions found in segmentation")
                return {
                    'lung_mask': lung_mask,
                    'lobe_mask': None,
                    'upper_lung': None,
                    'lower_lung': None,
                    'lobe_info': None
                }
            
            # Generate upper/lower lung masks using percentile-based division
            upper_lung, lower_lung, z_cut = self.divide_upper_lower_lung(lung_mask)
            
            logger.info(f"Basic lung segmentation completed: "
                       f"total={np.sum(lung_mask)}, "
                       f"upper={np.sum(upper_lung) if upper_lung is not None else 0}, "
                       f"lower={np.sum(lower_lung) if lower_lung is not None else 0} voxels, "
                       f"z_cut={z_cut}")
            
            return {
                'lung_mask': lung_mask,
                'lobe_mask': None,
                'upper_lung': upper_lung,
                'lower_lung': lower_lung,
                'lobe_info': None
            }
            
        except Exception as e:
            logger.error(f"Error in lung segmentation: {str(e)}")
            return {
                'lung_mask': np.zeros_like(ct_array, dtype=np.uint8),
                'lobe_mask': None,
                'upper_lung': None,
                'lower_lung': None,
                'lobe_info': {'error': str(e)}
            }
    
    def divide_upper_lower_lung(self, lung_mask: np.ndarray, 
                               percentile: float = 60) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Divide lung mask into upper and lower regions using percentile-based division
        
        Args:
            lung_mask: 3D binary lung mask
            percentile: Percentile for division (default: 60)
            
        Returns:
            Tuple of (upper_lung_mask, lower_lung_mask, z_cut_position)
        """
        # Use percentile-based division
        z_vox = np.where(lung_mask)[0]
        
        if len(z_vox) == 0:
            logger.warning("No lung pixels found in mask")
            return None, None, None
        
        z_cut = int(np.percentile(z_vox, percentile))
        
        upper_lung = lung_mask.copy()
        lower_lung = lung_mask.copy()
        
        upper_lung[:z_cut, :, :] = 0
        lower_lung[z_cut:, :, :] = 0
        
        logger.info(f"Used percentile-based division at z={z_cut}: "
                   f"upper={np.sum(upper_lung)} voxels, "
                   f"lower={np.sum(lower_lung)} voxels")
        
        return upper_lung, lower_lung, z_cut
    
    def calculate_volume(self, mask: np.ndarray, spacing: Tuple[float, float, float]) -> float:
        """
        Calculate volume from binary mask
        
        Args:
            mask: Binary mask
            spacing: Pixel spacing (z, y, x)
            
        Returns:
            Volume in cubic millimeters
        """
        if mask is None or not np.any(mask):
            return 0.0
        
        voxel_volume = spacing[0] * spacing[1] * spacing[2]
        voxel_count = np.sum(mask > 0)
        volume = voxel_count * voxel_volume
        
        return float(volume)
    
    def get_lung_statistics(self, lung_mask: np.ndarray, 
                           spacing: Tuple[float, float, float]) -> dict:
        """
        Get comprehensive lung statistics
        
        Args:
            lung_mask: 3D binary lung mask
            spacing: Pixel spacing
            
        Returns:
            Dictionary with lung statistics
        """
        stats = {
            "total_voxels": int(np.sum(lung_mask > 0)),
            "total_volume_mm3": self.calculate_volume(lung_mask, spacing),
            "shape": lung_mask.shape,
            "spacing": spacing,
            "method": "lungmask_percentile"
        }
        
        # Add upper/lower division statistics
        upper_lung, lower_lung, z_cut = self.divide_upper_lower_lung(lung_mask)
        
        if upper_lung is not None:
            stats.update({
                "z_cut_position": z_cut,
                "upper_voxels": int(np.sum(upper_lung > 0)),
                "lower_voxels": int(np.sum(lower_lung > 0)),
                "upper_volume_mm3": self.calculate_volume(upper_lung, spacing),
                "lower_volume_mm3": self.calculate_volume(lower_lung, spacing)
            })
        
        return stats
    

    
    def precompute_and_save(self, ct_array: np.ndarray, spacing: Tuple[float, float, float],
                           case_name: str, output_dir: str) -> dict:
        """
        Precompute lung segmentations and save to files (basic segmentation only)
        
        Args:
            ct_array: 3D CT array
            spacing: Pixel spacing
            case_name: Case identifier (e.g., "NSIP1")
            output_dir: Directory to save results
            
        Returns:
            Dictionary with file paths and statistics
        """
        try:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Perform basic segmentation
            result = self.segment_with_lobes(ct_array, spacing)
            
            # Prepare file paths (no lobe mask)
            files = {
                'lung_mask': os.path.join(output_dir, f"CT_{case_name}_lung_mask.nii.gz"),
                'upper_lung': os.path.join(output_dir, f"CT_{case_name}_upper_lung.nii.gz"),
                'lower_lung': os.path.join(output_dir, f"CT_{case_name}_lower_lung.nii.gz")
            }
            
            # Create reference image for proper spacing/orientation
            ref_sitk = sitk.GetImageFromArray(ct_array.astype(np.int16))
            ref_sitk.SetSpacing(spacing[::-1])
            
            # Save lung masks (no lobe mask)
            masks_to_save = [
                ('lung_mask', result['lung_mask']),
                ('upper_lung', result['upper_lung']),
                ('lower_lung', result['lower_lung'])
            ]
            
            saved_files = {}
            for name, mask in masks_to_save:
                if mask is not None:
                    mask_sitk = sitk.GetImageFromArray(mask.astype(np.uint8))
                    mask_sitk.CopyInformation(ref_sitk)
                    
                    sitk.WriteImage(mask_sitk, files[name])
                    saved_files[name] = files[name]
                    logger.debug(f"Saved {name}: {files[name]}")
            
            # Generate statistics
            stats = self.get_lung_statistics(result['lung_mask'], spacing)
            
            logger.info(f"Precomputed lung segmentation for {case_name}: "
                       f"{len(saved_files)} files saved (percentile-based division)")
            
            return {
                'files': saved_files,
                'statistics': stats,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Error precomputing segmentation for {case_name}: {str(e)}")
            return {
                'files': {},
                'statistics': {},
                'success': False,
                'error': str(e)
            }

# Compatibility alias to maintain existing interface
LungSegmentator = LungmaskSegmentator