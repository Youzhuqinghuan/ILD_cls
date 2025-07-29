"""
Lung segmentation module for ILD Analysis
Provides unified lung segmentation functionality
"""

import numpy as np
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import disk, binary_erosion, binary_closing
from skimage.filters import roberts
from scipy import ndimage as ndi
from typing import Tuple
import logging

logger = logging.getLogger(__name__)

class LungSegmentator:
    """Unified lung segmentation class"""
    
    def __init__(self, threshold: float = -300):
        """
        Initialize lung segmentator
        
        Args:
            threshold: HU threshold for lung segmentation
        """
        self.threshold = threshold
    
    def segment_slice(self, slice_2d: np.ndarray, spacing: Tuple[float, float, float]) -> np.ndarray:
        """
        Segment lung regions in a single 2D slice
        
        Args:
            slice_2d: 2D CT slice
            spacing: Pixel spacing (z, y, x)
            
        Returns:
            Binary mask of lung regions
        """
        # Step 1: Thresholding
        binary = slice_2d < self.threshold
        
        # Step 2: Clear border
        cleared = clear_border(binary)
        
        # Step 3: Connected component analysis
        label_image = label(cleared)
        
        # Keep only the two largest connected components (left and right lungs)
        areas = [r.area for r in regionprops(label_image)]
        areas.sort()
        
        if len(areas) > 2:
            for region in regionprops(label_image):
                if region.area < areas[-2]:
                    for coordinates in region.coords:
                        label_image[coordinates[0], coordinates[1]] = 0
        
        binary = label_image > 0
        
        # Step 4: Morphological operations
        if np.any(binary):
            # Erosion
            selem = disk(2)
            binary = binary_erosion(binary, selem)
            
            # Closing
            selem = disk(10)
            binary = binary_closing(binary, selem)
            
            # Fill holes
            edges = roberts(binary)
            binary = ndi.binary_fill_holes(edges)
        
        return binary.astype(np.uint8)
    
    def segment_volume(self, ct_array: np.ndarray, spacing: Tuple[float, float, float]) -> np.ndarray:
        """
        Segment lung regions in a 3D CT volume
        
        Args:
            ct_array: 3D CT array (Z, Y, X)
            spacing: Pixel spacing (z, y, x)
            
        Returns:
            3D binary mask of lung regions
        """
        lung_mask = np.zeros_like(ct_array, dtype=np.uint8)
        
        # Process each slice
        for i in range(ct_array.shape[0]):
            slice_mask = self.segment_slice(ct_array[i], spacing)
            lung_mask[i] = slice_mask
        
        logger.info(f"Segmented lung volume: {np.sum(lung_mask)} voxels")
        return lung_mask
    
    def divide_upper_lower_lung(self, lung_mask: np.ndarray, 
                               percentile: float = 60) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Divide lung mask into upper and lower regions
        
        Args:
            lung_mask: 3D binary lung mask
            percentile: Percentile for division (default 60%)
            
        Returns:
            Tuple of (upper_lung_mask, lower_lung_mask, z_cut_position)
        """
        # Get all lung voxel z-coordinates
        z_vox = np.where(lung_mask)[0]
        
        if len(z_vox) == 0:
            logger.warning("No lung pixels found in mask")
            return None, None, None
        
        # Calculate division point
        z_cut = int(np.percentile(z_vox, percentile))
        
        # Create upper and lower lung masks
        upper_lung = lung_mask.copy()
        lower_lung = lung_mask.copy()
        
        # Upper lung: keep parts above z_cut
        upper_lung[:z_cut, :, :] = 0
        
        # Lower lung: keep parts below z_cut  
        lower_lung[z_cut:, :, :] = 0
        
        logger.info(f"Divided lung at z={z_cut}: "
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
        
        # Calculate voxel volume
        voxel_volume = spacing[0] * spacing[1] * spacing[2]
        
        # Count voxels and calculate total volume
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
            "spacing": spacing
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