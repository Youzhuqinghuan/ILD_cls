"""
Lung distribution calculator (Variable 2)
Analyzes lesion distribution between upper and lower lungs
"""

import numpy as np
from typing import Dict, Tuple, Any
import logging

logger = logging.getLogger(__name__)

class LungDistributionCalculator:
    """Calculates lesion distribution between upper and lower lung regions"""
    
    def __init__(self, upper_lower_percentile: float = 60):
        """
        Initialize lung distribution calculator
        
        Args:
            upper_lower_percentile: Percentile for upper/lower lung division
        """
        self.upper_lower_percentile = upper_lower_percentile
    
    def divide_lung_regions(self, lung_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Divide lung mask into upper and lower regions
        
        Args:
            lung_mask: 3D binary lung mask
            
        Returns:
            Tuple of (upper_lung_mask, lower_lung_mask, z_cut_position)
        """
        # Get all lung voxel z-coordinates
        z_vox = np.where(lung_mask)[0]
        
        if len(z_vox) == 0:
            logger.warning("No lung pixels found in mask")
            return None, None, None
        
        # Calculate division point
        z_cut = int(np.percentile(z_vox, self.upper_lower_percentile))
        
        # Create upper and lower lung masks
        upper_lung = lung_mask.copy()
        lower_lung = lung_mask.copy()
        
        # Upper lung: keep parts above z_cut (higher z values = superior)
        upper_lung[:z_cut, :, :] = 0
        
        # Lower lung: keep parts below z_cut (lower z values = inferior)
        lower_lung[z_cut:, :, :] = 0
        
        return upper_lung, lower_lung, z_cut
    
    def calculate_lung_distribution(self, lesion_mask: np.ndarray, 
                                  lung_mask: np.ndarray,
                                  spacing: Tuple[float, float, float] = None) -> Dict[str, Any]:
        """
        Calculate lesion distribution between upper and lower lungs
        
        Args:
            lesion_mask: 3D lesion segmentation mask  
            lung_mask: 3D lung mask
            spacing: Pixel spacing (z, y, x)
            
        Returns:
            Dictionary with lung distribution analysis
        """
        results = {
            "upper_lung_volume": 0.0,
            "lower_lung_volume": 0.0,
            "total_lung_volume": 0.0,
            "upper_lesion_volume": 0.0,
            "lower_lesion_volume": 0.0,
            "total_lesion_volume": 0.0,
            "upper_lesion_ratio": 0.0,  # lesion volume / upper lung volume
            "lower_lesion_ratio": 0.0,  # lesion volume / lower lung volume
            "upper_lesion_proportion": 0.0,  # upper lesion / total lesion
            "lower_lesion_proportion": 0.0,  # lower lesion / total lesion
            "distribution_pattern": "unknown",
            "z_cut_position": None,
            "upper_lung_voxels": 0,
            "lower_lung_voxels": 0,
            "upper_lesion_voxels": 0,
            "lower_lesion_voxels": 0
        }
        
        try:
            # Divide lung into upper and lower regions
            upper_lung, lower_lung, z_cut = self.divide_lung_regions(lung_mask)
            
            if upper_lung is None:
                logger.error("Failed to divide lung regions")
                return results
            
            results["z_cut_position"] = z_cut
            
            # Calculate voxel volume
            voxel_volume = 1.0
            if spacing is not None:
                voxel_volume = spacing[0] * spacing[1] * spacing[2]
            
            # Ensure lesions are within lung boundaries
            lesion_in_lung = lesion_mask * (lung_mask > 0)
            
            # Calculate lesion distribution in upper and lower lungs
            lesion_in_upper = lesion_in_lung * upper_lung
            lesion_in_lower = lesion_in_lung * lower_lung
            
            # Count voxels
            upper_lung_voxels = np.sum(upper_lung > 0)
            lower_lung_voxels = np.sum(lower_lung > 0)
            upper_lesion_voxels = np.sum(lesion_in_upper > 0)
            lower_lesion_voxels = np.sum(lesion_in_lower > 0)
            total_lesion_voxels = np.sum(lesion_in_lung > 0)
            
            # Store voxel counts
            results["upper_lung_voxels"] = int(upper_lung_voxels)
            results["lower_lung_voxels"] = int(lower_lung_voxels)
            results["upper_lesion_voxels"] = int(upper_lesion_voxels)
            results["lower_lesion_voxels"] = int(lower_lesion_voxels)
            
            # Calculate volumes
            results["upper_lung_volume"] = float(upper_lung_voxels * voxel_volume)
            results["lower_lung_volume"] = float(lower_lung_voxels * voxel_volume)
            results["total_lung_volume"] = float((upper_lung_voxels + lower_lung_voxels) * voxel_volume)
            results["upper_lesion_volume"] = float(upper_lesion_voxels * voxel_volume)
            results["lower_lesion_volume"] = float(lower_lesion_voxels * voxel_volume)
            results["total_lesion_volume"] = float(total_lesion_voxels * voxel_volume)
            
            # Calculate ratios (lesion volume / lung volume in each region)
            if upper_lung_voxels > 0:
                results["upper_lesion_ratio"] = float(upper_lesion_voxels / upper_lung_voxels)
            
            if lower_lung_voxels > 0:
                results["lower_lesion_ratio"] = float(lower_lesion_voxels / lower_lung_voxels)
            
            # Calculate proportions (lesion distribution within total lesions)
            if total_lesion_voxels > 0:
                results["upper_lesion_proportion"] = float(upper_lesion_voxels / total_lesion_voxels)
                results["lower_lesion_proportion"] = float(lower_lesion_voxels / total_lesion_voxels)
            
            # Classify distribution pattern
            results["distribution_pattern"] = self._classify_distribution_pattern(
                results["upper_lesion_proportion"],
                results["lower_lesion_proportion"],
                results["upper_lesion_ratio"], 
                results["lower_lesion_ratio"]
            )
            
            logger.info(f"Lung distribution - Upper: {results['upper_lesion_proportion']:.2f}, "
                       f"Lower: {results['lower_lesion_proportion']:.2f}, "
                       f"Pattern: {results['distribution_pattern']}")
            
        except Exception as e:
            logger.error(f"Error calculating lung distribution: {str(e)}")
            results["error"] = str(e)
        
        return results
    
    def _classify_distribution_pattern(self, upper_proportion: float, lower_proportion: float,
                                     upper_ratio: float, lower_ratio: float) -> str:
        """
        Classify lesion distribution pattern
        
        Args:
            upper_proportion: Proportion of lesions in upper lung (0-1)
            lower_proportion: Proportion of lesions in lower lung (0-1) 
            upper_ratio: Lesion density in upper lung (0-1)
            lower_ratio: Lesion density in lower lung (0-1)
            
        Returns:
            Distribution pattern classification
        """
        # Thresholds for classification
        predominant_threshold = 0.65  # 65% threshold for predominance
        significant_threshold = 0.20   # 20% threshold for significant involvement
        
        try:
            if upper_proportion >= predominant_threshold:
                return "upper_predominant"
            elif lower_proportion >= predominant_threshold:
                return "lower_predominant"
            elif (upper_proportion >= significant_threshold and 
                  lower_proportion >= significant_threshold):
                # Check relative density to break ties
                if upper_ratio > lower_ratio * 1.5:
                    return "upper_predominant"
                elif lower_ratio > upper_ratio * 1.5:
                    return "lower_predominant"
                else:
                    return "diffuse"
            elif upper_proportion < significant_threshold and lower_proportion < significant_threshold:
                return "minimal"
            else:
                # One region has significant but not predominant involvement
                if upper_proportion > lower_proportion:
                    return "upper_predominant"
                else:
                    return "lower_predominant"
                    
        except Exception as e:
            logger.error(f"Error classifying distribution pattern: {str(e)}")
            return "unknown"
    
    def get_distribution_summary(self, distribution_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate summary of distribution analysis
        
        Args:
            distribution_results: Results from calculate_lung_distribution()
            
        Returns:
            Summary dictionary with key findings
        """
        summary = {
            "pattern": distribution_results.get("distribution_pattern", "unknown"),
            "upper_involvement": "none",
            "lower_involvement": "none", 
            "overall_severity": "minimal"
        }
        
        try:
            upper_ratio = distribution_results.get("upper_lesion_ratio", 0)
            lower_ratio = distribution_results.get("lower_lesion_ratio", 0)
            
            # Classify involvement levels
            def classify_involvement(ratio):
                if ratio < 0.05:  # < 5%
                    return "none"
                elif ratio < 0.15:  # 5-15%
                    return "mild"
                elif ratio < 0.30:  # 15-30%
                    return "moderate"
                else:  # > 30%
                    return "severe"
            
            summary["upper_involvement"] = classify_involvement(upper_ratio)
            summary["lower_involvement"] = classify_involvement(lower_ratio)
            
            # Overall severity based on maximum involvement
            max_ratio = max(upper_ratio, lower_ratio)
            summary["overall_severity"] = classify_involvement(max_ratio)
            
            # Additional context
            upper_prop = distribution_results.get("upper_lesion_proportion", 0)
            lower_prop = distribution_results.get("lower_lesion_proportion", 0)
            
            summary["upper_percentage"] = f"{upper_prop * 100:.1f}%"
            summary["lower_percentage"] = f"{lower_prop * 100:.1f}%"
            
        except Exception as e:
            logger.error(f"Error generating distribution summary: {str(e)}")
            summary["error"] = str(e)
        
        return summary