"""
Subpleural features calculator (Variable 4)
Analyzes subpleural involvement and characteristics
"""

import numpy as np
from scipy import ndimage as ndi
from typing import Dict, Tuple, Any
import logging

logger = logging.getLogger(__name__)

class SubpleuralFeaturesCalculator:
    """Calculates subpleural involvement and characteristics"""
    
    def __init__(self, subpleural_threshold_mm: float = 3.0):
        """
        Initialize subpleural features calculator
        
        Args:
            subpleural_threshold_mm: Distance threshold for subpleural classification (mm)
        """
        self.subpleural_threshold_mm = subpleural_threshold_mm
    
    def calculate_subpleural_features(self, lesion_mask: np.ndarray,
                                    lung_mask: np.ndarray,
                                    spacing: Tuple[float, float, float]) -> Dict[str, Any]:
        """
        Calculate subpleural involvement features
        
        Args:
            lesion_mask: 3D lesion segmentation mask
            lung_mask: 3D lung mask
            spacing: Pixel spacing (z, y, x)
            
        Returns:
            Dictionary with subpleural feature analysis
        """
        results = {
            "subpleural_involvement": False,
            "subpleural_features": "subpleural_omitted",
            "min_distance_to_pleura_mm": 0.0,
            "subpleural_lesion_ratio": 0.0,
            "subpleural_voxel_count": 0,
            "total_lesion_voxels": 0,
            "mean_subpleural_distance": 0.0,
            "subpleural_distribution": {},
            "pleural_contact_analysis": {}
        }
        
        try:
            # Calculate distance transform from lung boundary
            dist_transform = ndi.distance_transform_edt(lung_mask, sampling=spacing)
            
            # Ensure lesions are within lung boundaries
            lesion_in_lung = lesion_mask * (lung_mask > 0)
            
            # Get distances for all lesion voxels
            lesion_distances = dist_transform[lesion_in_lung > 0]
            
            if len(lesion_distances) == 0:
                logger.warning("No lesion voxels found within lung mask")
                return results
            
            results["total_lesion_voxels"] = int(len(lesion_distances))
            
            # Basic distance statistics
            min_distance = lesion_distances.min()
            results["min_distance_to_pleura_mm"] = float(min_distance)
            
            # Identify subpleural lesions
            subpleural_mask = lesion_distances <= self.subpleural_threshold_mm
            subpleural_count = np.sum(subpleural_mask)
            
            results["subpleural_voxel_count"] = int(subpleural_count)
            results["subpleural_lesion_ratio"] = float(subpleural_count / len(lesion_distances))
            
            # Determine subpleural involvement
            results["subpleural_involvement"] = min_distance <= self.subpleural_threshold_mm
            
            if results["subpleural_involvement"]:
                results["subpleural_features"] = "subpleural"
                
                # Calculate mean distance for subpleural lesions
                if subpleural_count > 0:
                    subpleural_distances = lesion_distances[subpleural_mask]
                    results["mean_subpleural_distance"] = float(subpleural_distances.mean())
            else:
                results["subpleural_features"] = "subpleural_omitted"
            
            # Detailed subpleural distribution analysis
            results["subpleural_distribution"] = self._analyze_subpleural_distribution(
                lesion_in_lung, dist_transform
            )
            
            # Pleural contact analysis
            results["pleural_contact_analysis"] = self._analyze_pleural_contact(
                lesion_in_lung, lung_mask, dist_transform
            )
            
            logger.info(f"Subpleural features - Involvement: {results['subpleural_involvement']}, "
                       f"Min distance: {results['min_distance_to_pleura_mm']:.2f}mm, "
                       f"Ratio: {results['subpleural_lesion_ratio']:.2f}")
            
        except Exception as e:
            logger.error(f"Error calculating subpleural features: {str(e)}")
            results["error"] = str(e)
        
        return results
    
    def _analyze_subpleural_distribution(self, lesion_mask: np.ndarray,
                                       dist_transform: np.ndarray) -> Dict[str, Any]:
        """
        Analyze distribution of subpleural lesions
        
        Args:
            lesion_mask: Lesion mask within lung boundaries
            dist_transform: Distance transform from lung boundary
            
        Returns:
            Dictionary with subpleural distribution analysis
        """
        distribution = {
            "distance_0_1mm": {"count": 0, "percentage": 0.0},
            "distance_1_2mm": {"count": 0, "percentage": 0.0},
            "distance_2_3mm": {"count": 0, "percentage": 0.0},
            "distance_3_5mm": {"count": 0, "percentage": 0.0},
            "distance_5mm_plus": {"count": 0, "percentage": 0.0}
        }
        
        try:
            lesion_distances = dist_transform[lesion_mask > 0]
            total_lesions = len(lesion_distances)
            
            if total_lesions == 0:
                return distribution
            
            # Define distance ranges
            ranges = [
                ("distance_0_1mm", 0, 1),
                ("distance_1_2mm", 1, 2),
                ("distance_2_3mm", 2, 3),
                ("distance_3_5mm", 3, 5),
                ("distance_5mm_plus", 5, float('inf'))
            ]
            
            for range_name, min_dist, max_dist in ranges:
                if max_dist == float('inf'):
                    count = np.sum(lesion_distances >= min_dist)
                else:
                    count = np.sum((lesion_distances >= min_dist) & (lesion_distances < max_dist))
                
                distribution[range_name]["count"] = int(count)
                distribution[range_name]["percentage"] = float(count / total_lesions * 100)
            
        except Exception as e:
            logger.error(f"Error analyzing subpleural distribution: {str(e)}")
            distribution["error"] = str(e)
        
        return distribution
    
    def _analyze_pleural_contact(self, lesion_mask: np.ndarray,
                               lung_mask: np.ndarray,
                               dist_transform: np.ndarray) -> Dict[str, Any]:
        """
        Analyze direct pleural contact characteristics
        
        Args:
            lesion_mask: Lesion mask within lung boundaries
            lung_mask: Full lung mask
            dist_transform: Distance transform from lung boundary
            
        Returns:
            Dictionary with pleural contact analysis
        """
        contact_analysis = {
            "direct_pleural_contact": False,
            "contact_surface_area": 0,
            "contact_percentage": 0.0,
            "contact_regions": 0
        }
        
        try:
            # Identify lung boundary (distance = 0 or very close to boundary)
            boundary_tolerance = 0.5  # mm
            boundary_mask = dist_transform <= boundary_tolerance
            
            # Find lesions in direct contact with pleura
            pleural_contact_mask = (lesion_mask > 0) & boundary_mask
            contact_voxels = np.sum(pleural_contact_mask)
            
            contact_analysis["direct_pleural_contact"] = contact_voxels > 0
            contact_analysis["contact_surface_area"] = int(contact_voxels)
            
            # Calculate contact percentage
            total_lesion_voxels = np.sum(lesion_mask > 0)
            if total_lesion_voxels > 0:
                contact_analysis["contact_percentage"] = float(
                    contact_voxels / total_lesion_voxels * 100
                )
            
            # Count separate contact regions (connected components)
            if contact_voxels > 0:
                from skimage.measure import label
                labeled_contacts = label(pleural_contact_mask)
                contact_analysis["contact_regions"] = int(labeled_contacts.max())
            
        except Exception as e:
            logger.error(f"Error analyzing pleural contact: {str(e)}")
            contact_analysis["error"] = str(e)
        
        return contact_analysis
    
    def classify_subpleural_pattern(self, subpleural_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify subpleural pattern based on characteristics
        
        Args:
            subpleural_results: Results from calculate_subpleural_features()
            
        Returns:
            Dictionary with pattern classification
        """
        classification = {
            "pattern_type": "no_subpleural_involvement",
            "severity": "none",
            "characteristics": [],
            "uip_likelihood": 0.0,
            "nsip_likelihood": 0.0
        }
        
        try:
            involvement = subpleural_results.get("subpleural_involvement", False)
            ratio = subpleural_results.get("subpleural_lesion_ratio", 0.0)
            min_distance = subpleural_results.get("min_distance_to_pleura_mm", 0.0)
            
            if not involvement:
                classification["pattern_type"] = "no_subpleural_involvement"
                classification["nsip_likelihood"] = 0.6  # NSIP often spares subpleural
                return classification
            
            # Classify based on subpleural involvement ratio
            if ratio >= 0.5:  # ≥50% subpleural
                classification["pattern_type"] = "extensive_subpleural"
                classification["severity"] = "severe"
                classification["uip_likelihood"] = 0.8
                classification["characteristics"].append("extensive_subpleural_involvement")
            elif ratio >= 0.2:  # 20-50% subpleural
                classification["pattern_type"] = "moderate_subpleural"
                classification["severity"] = "moderate"
                classification["uip_likelihood"] = 0.6
                classification["characteristics"].append("moderate_subpleural_involvement")
            else:  # <20% subpleural
                classification["pattern_type"] = "minimal_subpleural"
                classification["severity"] = "mild"
                classification["uip_likelihood"] = 0.3
                classification["nsip_likelihood"] = 0.4
                classification["characteristics"].append("minimal_subpleural_involvement")
            
            # Additional characteristics based on contact analysis
            contact_analysis = subpleural_results.get("pleural_contact_analysis", {})
            if contact_analysis.get("direct_pleural_contact", False):
                classification["characteristics"].append("direct_pleural_contact")
                classification["uip_likelihood"] += 0.1
            
            # Distance-based characteristics
            if min_distance <= 1.0:
                classification["characteristics"].append("immediate_subpleural")
                classification["uip_likelihood"] += 0.1
            
            # Distribution analysis
            distribution = subpleural_results.get("subpleural_distribution", {})
            very_close_percentage = (
                distribution.get("distance_0_1mm", {}).get("percentage", 0) +
                distribution.get("distance_1_2mm", {}).get("percentage", 0)
            )
            
            if very_close_percentage >= 15:  # ≥15% within 2mm
                classification["characteristics"].append("predominantly_immediate_subpleural")
                classification["uip_likelihood"] += 0.1
            
            # Normalize likelihoods
            classification["uip_likelihood"] = min(1.0, classification["uip_likelihood"])
            classification["nsip_likelihood"] = max(0.0, 1.0 - classification["uip_likelihood"])
            
        except Exception as e:
            logger.error(f"Error classifying subpleural pattern: {str(e)}")
            classification["error"] = str(e)
        
        return classification
    
    def get_subpleural_summary(self, subpleural_results: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate concise summary of subpleural findings
        
        Args:
            subpleural_results: Results from calculate_subpleural_features()
            
        Returns:
            Summary dictionary with key findings in readable format
        """
        summary = {
            "involvement": "No",
            "severity": "None",
            "min_distance": "N/A",
            "distribution": "N/A"
        }
        
        try:
            if subpleural_results.get("subpleural_involvement", False):
                summary["involvement"] = "Yes"
                
                ratio = subpleural_results.get("subpleural_lesion_ratio", 0.0)
                if ratio >= 0.5:
                    summary["severity"] = "Extensive (≥50%)"
                elif ratio >= 0.2:
                    summary["severity"] = "Moderate (20-50%)"
                else:
                    summary["severity"] = "Minimal (<20%)"
                
                min_dist = subpleural_results.get("min_distance_to_pleura_mm", 0.0)
                summary["min_distance"] = f"{min_dist:.1f}mm"
                
                # Predominant distance range
                distribution = subpleural_results.get("subpleural_distribution", {})
                max_percentage = 0
                max_range = ""
                
                for range_name, data in distribution.items():
                    if isinstance(data, dict) and "percentage" in data:
                        percentage = data["percentage"]
                        if percentage > max_percentage:
                            max_percentage = percentage
                            max_range = range_name.replace("distance_", "").replace("_", "-")
                
                if max_range:
                    summary["distribution"] = f"Predominantly {max_range}mm from pleura"
            
        except Exception as e:
            logger.error(f"Error generating subpleural summary: {str(e)}")
            summary["error"] = str(e)
        
        return summary