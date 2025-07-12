"""
Axial distribution calculator (Variable 3)
Analyzes lesion distribution in axial plane: peripheral/central/diffuse
"""

import numpy as np
from scipy import ndimage as ndi
from typing import Dict, Tuple, Any
import logging

logger = logging.getLogger(__name__)

class AxialDistributionCalculator:
    """Calculates lesion distribution in axial plane (peripheral vs central)"""
    
    def __init__(self, peripheral_threshold_mm: float = 10.0, 
                 peripheral_ratio_threshold: float = 0.20,
                 classification_threshold: float = 0.70):
        """
        Initialize axial distribution calculator
        
        Args:
            peripheral_threshold_mm: Distance threshold for peripheral classification (mm)
            peripheral_ratio_threshold: Relative distance threshold for peripheral
            classification_threshold: Threshold for peripheral/central classification
        """
        self.peripheral_threshold_mm = peripheral_threshold_mm
        self.peripheral_ratio_threshold = peripheral_ratio_threshold
        self.classification_threshold = classification_threshold
    
    def calculate_distance_transform(self, lung_mask: np.ndarray, 
                                   spacing: Tuple[float, float, float]) -> np.ndarray:
        """
        Calculate distance transform from lung boundary
        
        Args:
            lung_mask: 3D binary lung mask
            spacing: Pixel spacing (z, y, x)
            
        Returns:
            Distance transform array (distance to lung boundary in mm)
        """
        try:
            # Calculate Euclidean distance transform
            dist_transform = ndi.distance_transform_edt(lung_mask, sampling=spacing)
            return dist_transform
            
        except Exception as e:
            logger.error(f"Error calculating distance transform: {str(e)}")
            return None
    
    def calculate_axial_distribution(self, lesion_mask: np.ndarray,
                                   lung_mask: np.ndarray,
                                   spacing: Tuple[float, float, float]) -> Dict[str, Any]:
        """
        Calculate axial distribution of lesions
        
        Args:
            lesion_mask: 3D lesion segmentation mask
            lung_mask: 3D lung mask 
            spacing: Pixel spacing (z, y, x)
            
        Returns:
            Dictionary with axial distribution analysis
        """
        results = {
            "axial_distribution": "unknown",
            "peripheral_ratio": 0.0,
            "central_ratio": 0.0,
            "mean_distance_mm": 0.0,
            "min_distance_mm": 0.0,
            "max_distance_mm": 0.0,
            "max_radius_mm": 0.0,
            "peripheral_voxels": 0,
            "central_voxels": 0,
            "total_lesion_voxels": 0
        }
        
        try:
            # Calculate distance transform
            dist_transform = self.calculate_distance_transform(lung_mask, spacing)
            if dist_transform is None:
                return results
            
            # Ensure lesions are within lung boundaries
            lesion_in_lung = lesion_mask * (lung_mask > 0)
            
            # Get distances for lesion voxels
            lesion_distances = dist_transform[lesion_in_lung > 0]
            
            if len(lesion_distances) == 0:
                logger.warning("No lesion voxels found within lung mask")
                return results
            
            # Calculate maximum radius for normalization
            max_radius = dist_transform.max()
            results["max_radius_mm"] = float(max_radius)
            
            # Basic distance statistics
            results["mean_distance_mm"] = float(lesion_distances.mean())
            results["min_distance_mm"] = float(lesion_distances.min())
            results["max_distance_mm"] = float(lesion_distances.max())
            results["total_lesion_voxels"] = int(len(lesion_distances))
            
            # Classify voxels as peripheral or central
            peripheral_mask = self._classify_peripheral_voxels(
                lesion_distances, max_radius
            )
            central_mask = ~peripheral_mask
            
            # Count peripheral and central voxels
            peripheral_count = np.sum(peripheral_mask)
            central_count = np.sum(central_mask)
            total_count = len(lesion_distances)
            
            results["peripheral_voxels"] = int(peripheral_count)
            results["central_voxels"] = int(central_count)
            
            # Calculate ratios
            if total_count > 0:
                results["peripheral_ratio"] = float(peripheral_count / total_count)
                results["central_ratio"] = float(central_count / total_count)
            
            # Classify overall axial distribution
            results["axial_distribution"] = self._classify_axial_distribution(
                results["peripheral_ratio"], results["central_ratio"]
            )
            
            logger.info(f"Axial distribution - Peripheral: {results['peripheral_ratio']:.2f}, "
                       f"Central: {results['central_ratio']:.2f}, "
                       f"Pattern: {results['axial_distribution']}")
            
        except Exception as e:
            logger.error(f"Error calculating axial distribution: {str(e)}")
            results["error"] = str(e)
        
        return results
    
    def _classify_peripheral_voxels(self, distances: np.ndarray, max_radius: float) -> np.ndarray:
        """
        Classify voxels as peripheral based on distance criteria
        
        Args:
            distances: Array of distances to lung boundary
            max_radius: Maximum distance (lung center to boundary)
            
        Returns:
            Boolean array indicating peripheral voxels
        """
        try:
            # Two criteria for peripheral classification:
            # 1. Absolute distance <= threshold (mm)
            # 2. Relative distance <= threshold * max_radius
            
            absolute_criterion = distances <= self.peripheral_threshold_mm
            relative_criterion = distances <= (self.peripheral_ratio_threshold * max_radius)
            
            # A voxel is peripheral if it meets EITHER criterion
            peripheral_mask = absolute_criterion | relative_criterion
            
            return peripheral_mask
            
        except Exception as e:
            logger.error(f"Error classifying peripheral voxels: {str(e)}")
            return np.zeros_like(distances, dtype=bool)
    
    def _classify_axial_distribution(self, peripheral_ratio: float, central_ratio: float) -> str:
        """
        Classify overall axial distribution pattern
        
        Args:
            peripheral_ratio: Ratio of peripheral lesions (0-1)
            central_ratio: Ratio of central lesions (0-1)
            
        Returns:
            Axial distribution classification
        """
        try:
            if peripheral_ratio >= self.classification_threshold:
                return "peripheral"
            elif central_ratio >= self.classification_threshold:
                return "central"
            else:
                return "diffuse_scattered"
                
        except Exception as e:
            logger.error(f"Error classifying axial distribution: {str(e)}")
            return "unknown"
    
    def analyze_regional_distribution(self, lesion_mask: np.ndarray,
                                    lung_mask: np.ndarray,
                                    spacing: Tuple[float, float, float]) -> Dict[str, Any]:
        """
        Analyze lesion distribution in different lung regions
        
        Args:
            lesion_mask: 3D lesion segmentation mask
            lung_mask: 3D lung mask
            spacing: Pixel spacing
            
        Returns:
            Dictionary with regional distribution analysis
        """
        regional_results = {
            "slice_wise_distribution": [],
            "zone_analysis": {},
            "gradient_analysis": {}
        }
        
        try:
            # Calculate distance transform once
            dist_transform = self.calculate_distance_transform(lung_mask, spacing)
            if dist_transform is None:
                return regional_results
            
            lesion_in_lung = lesion_mask * (lung_mask > 0)
            
            # Analyze slice by slice
            for z in range(lesion_mask.shape[0]):
                lung_slice = lung_mask[z]
                lesion_slice = lesion_in_lung[z]
                
                if np.any(lung_slice) and np.any(lesion_slice):
                    slice_distances = dist_transform[z][lesion_slice > 0]
                    
                    if len(slice_distances) > 0:
                        max_radius_slice = dist_transform[z].max()
                        peripheral_mask = self._classify_peripheral_voxels(
                            slice_distances, max_radius_slice
                        )
                        
                        slice_result = {
                            "slice_index": z,
                            "total_lesion_voxels": len(slice_distances),
                            "peripheral_ratio": float(np.mean(peripheral_mask)),
                            "mean_distance": float(slice_distances.mean())
                        }
                        regional_results["slice_wise_distribution"].append(slice_result)
            
            # Zone-based analysis (divide lung into concentric zones)
            regional_results["zone_analysis"] = self._analyze_concentric_zones(
                lesion_in_lung, dist_transform
            )
            
            # Gradient analysis (how lesion density changes with distance)
            regional_results["gradient_analysis"] = self._analyze_distance_gradient(
                lesion_in_lung, dist_transform
            )
            
        except Exception as e:
            logger.error(f"Error in regional distribution analysis: {str(e)}")
            regional_results["error"] = str(e)
        
        return regional_results
    
    def _analyze_concentric_zones(self, lesion_mask: np.ndarray, 
                                dist_transform: np.ndarray) -> Dict[str, Any]:
        """Analyze lesion distribution in concentric zones"""
        zone_analysis = {
            "zone_1_peripheral": {"distance_range": "0-5mm", "lesion_count": 0, "lung_volume": 0},
            "zone_2_intermediate": {"distance_range": "5-15mm", "lesion_count": 0, "lung_volume": 0},
            "zone_3_central": {"distance_range": ">15mm", "lesion_count": 0, "lung_volume": 0}
        }
        
        try:
            # Define zones based on distance thresholds
            zone_1_mask = (dist_transform > 0) & (dist_transform <= 5)
            zone_2_mask = (dist_transform > 5) & (dist_transform <= 15)
            zone_3_mask = (dist_transform > 15)
            
            # Count lesions and lung volume in each zone
            for zone_name, zone_mask in [
                ("zone_1_peripheral", zone_1_mask),
                ("zone_2_intermediate", zone_2_mask), 
                ("zone_3_central", zone_3_mask)
            ]:
                lesions_in_zone = np.sum((lesion_mask > 0) & zone_mask)
                lung_in_zone = np.sum(zone_mask)
                
                zone_analysis[zone_name]["lesion_count"] = int(lesions_in_zone)
                zone_analysis[zone_name]["lung_volume"] = int(lung_in_zone)
                
                if lung_in_zone > 0:
                    zone_analysis[zone_name]["lesion_density"] = float(lesions_in_zone / lung_in_zone)
                else:
                    zone_analysis[zone_name]["lesion_density"] = 0.0
                    
        except Exception as e:
            logger.error(f"Error in concentric zone analysis: {str(e)}")
            zone_analysis["error"] = str(e)
        
        return zone_analysis
    
    def _analyze_distance_gradient(self, lesion_mask: np.ndarray,
                                 dist_transform: np.ndarray) -> Dict[str, Any]:
        """Analyze how lesion density changes with distance from pleura"""
        gradient_analysis = {
            "distance_bins": [],
            "lesion_densities": [],
            "trend": "unknown"
        }
        
        try:
            # Create distance bins
            max_distance = dist_transform.max()
            bin_edges = np.linspace(0, max_distance, 11)  # 10 bins
            
            densities = []
            bin_centers = []
            
            for i in range(len(bin_edges) - 1):
                bin_mask = (dist_transform >= bin_edges[i]) & (dist_transform < bin_edges[i + 1])
                lung_voxels_in_bin = np.sum(bin_mask)
                lesion_voxels_in_bin = np.sum((lesion_mask > 0) & bin_mask)
                
                bin_center = (bin_edges[i] + bin_edges[i + 1]) / 2
                density = lesion_voxels_in_bin / lung_voxels_in_bin if lung_voxels_in_bin > 0 else 0
                
                bin_centers.append(float(bin_center))
                densities.append(float(density))
            
            gradient_analysis["distance_bins"] = bin_centers
            gradient_analysis["lesion_densities"] = densities
            
            # Determine trend (increasing, decreasing, or flat)
            if len(densities) > 1:
                correlation = np.corrcoef(bin_centers, densities)[0, 1]
                if correlation > 0.3:
                    gradient_analysis["trend"] = "increasing_centrally"
                elif correlation < -0.3:
                    gradient_analysis["trend"] = "decreasing_centrally"
                else:
                    gradient_analysis["trend"] = "uniform"
            
        except Exception as e:
            logger.error(f"Error in distance gradient analysis: {str(e)}")
            gradient_analysis["error"] = str(e)
        
        return gradient_analysis