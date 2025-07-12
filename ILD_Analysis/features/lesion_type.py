"""
Lesion type extractor (Variable 1)
Analyzes lesion predictions to determine dominant lesion types
"""

import numpy as np
from typing import Dict, List, Tuple, Any
import logging

logger = logging.getLogger(__name__)

class LesionTypeExtractor:
    """Extracts lesion type information from predicted segmentation masks"""
    
    def __init__(self, lesion_labels: Dict[int, str] = None):
        """
        Initialize lesion type extractor
        
        Args:
            lesion_labels: Mapping from label values to lesion names
        """
        self.lesion_labels = lesion_labels or {
            0: "background",
            1: "honeycomb",
            2: "reticulation", 
            3: "ground_glass_opacity",
            4: "consolidation"
        }
    
    def extract_lesion_types(self, lesion_mask: np.ndarray, 
                           spacing: Tuple[float, float, float] = None) -> Dict[str, Any]:
        """
        Extract lesion type information from segmentation mask
        
        Args:
            lesion_mask: 3D lesion segmentation mask with label values
            spacing: Pixel spacing (z, y, x)
            
        Returns:
            Dictionary with lesion type analysis results
        """
        results = {
            "lesion_types_present": [],
            "lesion_volumes": {},
            "lesion_voxel_counts": {},
            "lesion_percentages": {},
            "dominant_lesion": None,
            "total_lesion_volume": 0.0,
            "total_lesion_voxels": 0
        }
        
        try:
            # Get unique labels (excluding background)
            unique_labels = np.unique(lesion_mask)
            lesion_labels = unique_labels[unique_labels > 0]  # Exclude background (0)
            
            if len(lesion_labels) == 0:
                logger.warning("No lesion labels found in mask")
                return results
            
            # Calculate voxel volume if spacing is provided
            voxel_volume = 1.0
            if spacing is not None:
                voxel_volume = spacing[0] * spacing[1] * spacing[2]
            
            total_lesion_voxels = 0
            lesion_counts = {}
            
            # Count voxels for each lesion type
            for label_val in lesion_labels:
                if label_val in self.lesion_labels:
                    lesion_name = self.lesion_labels[label_val]
                    voxel_count = np.sum(lesion_mask == label_val)
                    volume = voxel_count * voxel_volume
                    
                    lesion_counts[lesion_name] = voxel_count
                    results["lesion_voxel_counts"][lesion_name] = int(voxel_count)
                    results["lesion_volumes"][lesion_name] = float(volume)
                    results["lesion_types_present"].append(lesion_name)
                    
                    total_lesion_voxels += voxel_count
                else:
                    logger.warning(f"Unknown lesion label: {label_val}")
            
            results["total_lesion_voxels"] = int(total_lesion_voxels)
            results["total_lesion_volume"] = float(total_lesion_voxels * voxel_volume)
            
            # Calculate percentages and find dominant lesion
            if total_lesion_voxels > 0:
                max_count = 0
                dominant_lesion = None
                
                for lesion_name, count in lesion_counts.items():
                    percentage = (count / total_lesion_voxels) * 100
                    results["lesion_percentages"][lesion_name] = float(percentage)
                    
                    if count > max_count:
                        max_count = count
                        dominant_lesion = lesion_name
                
                results["dominant_lesion"] = dominant_lesion
            
            logger.info(f"Extracted lesion types: {results['lesion_types_present']}")
            logger.info(f"Dominant lesion: {results['dominant_lesion']}")
            
        except Exception as e:
            logger.error(f"Error extracting lesion types: {str(e)}")
            results["error"] = str(e)
        
        return results
    
    def classify_lesion_pattern(self, lesion_types: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify overall lesion pattern based on present lesion types
        
        Args:
            lesion_types: Results from extract_lesion_types()
            
        Returns:
            Dictionary with pattern classification
        """
        classification = {
            "pattern_category": "unknown",
            "confidence": 0.0,
            "reasoning": []
        }
        
        try:
            present_lesions = lesion_types.get("lesion_types_present", [])
            dominant_lesion = lesion_types.get("dominant_lesion")
            percentages = lesion_types.get("lesion_percentages", {})
            
            if not present_lesions:
                classification["pattern_category"] = "normal"
                classification["reasoning"].append("No lesions detected")
                return classification
            
            # Classification rules based on lesion types
            if "honeycomb" in present_lesions:
                # Presence of honeycomb suggests UIP pattern
                classification["pattern_category"] = "uip_like"
                classification["confidence"] = 0.8
                classification["reasoning"].append("Honeycomb pattern present")
                
                if dominant_lesion == "honeycomb":
                    classification["confidence"] = 0.9
                    classification["reasoning"].append("Honeycomb is dominant lesion")
            
            elif dominant_lesion == "ground_glass_opacity":
                ggo_percentage = percentages.get("ground_glass_opacity", 0)
                
                if "consolidation" in present_lesions:
                    # GGO + consolidation suggests OP
                    classification["pattern_category"] = "op_like"
                    classification["confidence"] = 0.7
                    classification["reasoning"].append("GGO with consolidation present")
                else:
                    # Pure GGO or GGO + reticulation suggests NSIP
                    classification["pattern_category"] = "nsip_like"
                    classification["confidence"] = 0.6
                    classification["reasoning"].append("Ground glass opacity dominant")
                    
                    if "reticulation" in present_lesions:
                        classification["confidence"] = 0.7
                        classification["reasoning"].append("GGO with reticulation")
            
            elif dominant_lesion == "reticulation":
                if "ground_glass_opacity" in present_lesions:
                    classification["pattern_category"] = "nsip_like"
                    classification["confidence"] = 0.7
                    classification["reasoning"].append("Reticulation with GGO")
                else:
                    classification["pattern_category"] = "fibrotic"
                    classification["confidence"] = 0.6
                    classification["reasoning"].append("Reticulation dominant")
            
            elif dominant_lesion == "consolidation":
                classification["pattern_category"] = "op_like"
                classification["confidence"] = 0.6
                classification["reasoning"].append("Consolidation dominant")
            
            else:
                classification["pattern_category"] = "other"
                classification["confidence"] = 0.3
                classification["reasoning"].append("Mixed or unclear pattern")
            
            logger.info(f"Lesion pattern classification: {classification['pattern_category']} "
                       f"(confidence: {classification['confidence']:.2f})")
            
        except Exception as e:
            logger.error(f"Error classifying lesion pattern: {str(e)}")
            classification["error"] = str(e)
        
        return classification
    
    def get_lesion_distribution(self, lesion_mask: np.ndarray, 
                               lung_mask: np.ndarray) -> Dict[str, Any]:
        """
        Analyze spatial distribution of lesions within lungs
        
        Args:
            lesion_mask: 3D lesion segmentation mask
            lung_mask: 3D lung mask
            
        Returns:
            Dictionary with spatial distribution analysis
        """
        distribution = {
            "total_lung_voxels": 0,
            "total_lesion_voxels": 0,
            "lesion_lung_ratio": 0.0,
            "coverage_percentage": 0.0
        }
        
        try:
            # Ensure lesions are within lung boundaries
            lesion_in_lung = lesion_mask * (lung_mask > 0)
            
            total_lung_voxels = np.sum(lung_mask > 0)
            total_lesion_voxels = np.sum(lesion_in_lung > 0)
            
            distribution["total_lung_voxels"] = int(total_lung_voxels)
            distribution["total_lesion_voxels"] = int(total_lesion_voxels)
            
            if total_lung_voxels > 0:
                ratio = total_lesion_voxels / total_lung_voxels
                distribution["lesion_lung_ratio"] = float(ratio)
                distribution["coverage_percentage"] = float(ratio * 100)
            
            logger.info(f"Lesion coverage: {distribution['coverage_percentage']:.2f}% of lung volume")
            
        except Exception as e:
            logger.error(f"Error analyzing lesion distribution: {str(e)}")
            distribution["error"] = str(e)
        
        return distribution