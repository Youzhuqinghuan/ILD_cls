"""
Full pipeline for ILD Analysis
Orchestrates the complete workflow from CT images and lesion predictions to pattern classification
"""

import os
import json
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

# Import all components
from ..core.data_loader import ILDDataLoader
from ..core.lung_segmentation_legacy import LungSegmentator as LegacyLungSegmentator
from ..core.lungmask_segmentation import LungmaskSegmentator
from ..features.lesion_type import LesionTypeExtractor
from ..features.lung_distribution import LungDistributionCalculator
from ..features.axial_distribution import AxialDistributionCalculator
from ..features.subpleural_features import SubpleuralFeaturesCalculator
from ..classification.pattern_classifier import ILDPatternClassifier
from ..config import Config

logger = logging.getLogger(__name__)

class ILDAnalysisPipeline:
    """Complete ILD analysis pipeline"""
    
    def __init__(self, ct_dir: str = None, lesion_dir: str = None, 
                 output_dir: str = None, lung_mask_dir: str = None,
                 use_precomputed_masks: bool = True):
        """
        Initialize ILD Analysis Pipeline
        
        Args:
            ct_dir: Directory containing CT images (default from config)
            lesion_dir: Directory containing lesion predictions (default from config)
            output_dir: Directory for output results (default from config)
            lung_mask_dir: Directory containing precomputed lung masks (default: dataset/lungs)
            use_precomputed_masks: Whether to use precomputed masks when available
        """
        # Set directories
        self.ct_dir = ct_dir or Config.CT_IMAGES_DIR
        self.lesion_dir = lesion_dir or Config.PREDICTED_LABELS_DIR
        self.output_dir = output_dir or Config.OUTPUT_DIR
        
        # Set lung mask directory (default to dataset/lungs)
        if lung_mask_dir is None:
            # Assume lung masks are in dataset/lungs relative to ct_dir
            dataset_dir = os.path.dirname(self.ct_dir)
            self.lung_mask_dir = os.path.join(dataset_dir, 'lungs')
        else:
            self.lung_mask_dir = lung_mask_dir
            
        self.use_precomputed_masks = use_precomputed_masks
        
        # Initialize components
        self.data_loader = ILDDataLoader(self.ct_dir, self.lesion_dir, self.lung_mask_dir)
        
        # Initialize lung segmentator (try lungmask first, fallback to legacy)
        try:
            self.lung_segmentator = LungmaskSegmentator(enable_lobe_segmentation=False)
            self.segmentation_method = "lungmask"
            logger.info("Using lungmask-based segmentation (percentile-based upper/lower division)")
        except Exception as e:
            logger.warning(f"Failed to initialize lungmask: {str(e)}")
            logger.info("Falling back to legacy morphological segmentation")
            self.lung_segmentator = LegacyLungSegmentator(threshold=Config.LUNG_THRESHOLD)
            self.segmentation_method = "legacy"
        self.lesion_extractor = LesionTypeExtractor(Config.LESION_LABELS)
        self.lung_distribution_calc = LungDistributionCalculator(Config.UPPER_LOWER_PERCENTILE)
        self.axial_distribution_calc = AxialDistributionCalculator(
            Config.PERIPHERAL_THRESHOLD_MM,
            Config.PERIPHERAL_RATIO_THRESHOLD,
            Config.AXIAL_CLASSIFICATION_THRESHOLD
        )
        self.subpleural_calc = SubpleuralFeaturesCalculator(Config.SUBPLEURAL_THRESHOLD_MM)
        self.pattern_classifier = ILDPatternClassifier()
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
    
    def process_single_case(self, case_name: str, save_intermediate: bool = False) -> Dict[str, Any]:
        """
        Process a single case through the complete pipeline
        
        Args:
            case_name: Case identifier (e.g., "NSIP1")
            save_intermediate: Whether to save intermediate results
            
        Returns:
            Dictionary with complete analysis results
        """
        result = {
            "case_name": case_name,
            "success": False,
            "timestamp": datetime.now().isoformat(),
            "error": None
        }
        
        try:
            logger.info(f"Processing case: {case_name}")
            
            # Step 1: Load data (with potential precomputed lung masks)
            ct_array, lesion_array, spacing, ct_sitk, precomputed_masks = self.data_loader.load_case_with_lung_masks(case_name)
            
            if ct_array is None:
                result["error"] = "Failed to load case data"
                return result
            
            result["data_info"] = {
                "ct_shape": ct_array.shape,
                "lesion_shape": lesion_array.shape,
                "spacing": spacing,
                "precomputed_masks_available": any(mask is not None for mask in precomputed_masks.values()),
                "segmentation_method": self.segmentation_method
            }
            
            # Step 2: Lung segmentation (use precomputed if available)
            logger.info(f"Obtaining lung segmentation for {case_name}")
            
            if (self.use_precomputed_masks and 
                precomputed_masks.get('lung_mask') is not None):
                # Use precomputed lung mask
                lung_mask = precomputed_masks['lung_mask']
                upper_lung = precomputed_masks.get('upper_lung')
                lower_lung = precomputed_masks.get('lower_lung')
                
                logger.info(f"Using precomputed lung masks for {case_name}")
                result["data_info"]["lung_segmentation_source"] = "precomputed"
                
            else:
                # Compute lung segmentation in real-time
                logger.info(f"Computing lung segmentation in real-time for {case_name}")
                lung_mask = self.lung_segmentator.segment_volume(ct_array, spacing)
                
                # Get upper/lower division
                if hasattr(self.lung_segmentator, 'divide_upper_lower_lung'):
                    upper_lung, lower_lung, _ = self.lung_segmentator.divide_upper_lower_lung(lung_mask)
                else:
                    upper_lung = lower_lung = None
                
                result["data_info"]["lung_segmentation_source"] = "real_time"
            
            if not np.any(lung_mask):
                result["error"] = "Failed to segment lungs"
                return result
            
            # Step 3: Extract four variables
            logger.info(f"Extracting features for {case_name}")
            
            # Variable 1: Lesion type
            lesion_features = self.lesion_extractor.extract_lesion_types(
                lesion_array, spacing
            )
            lesion_pattern = self.lesion_extractor.classify_lesion_pattern(lesion_features)
            lesion_distribution = self.lesion_extractor.get_lesion_distribution(
                lesion_array, lung_mask
            )
            
            # Variable 2: Lung distribution (upper/lower)
            if upper_lung is not None and lower_lung is not None:
                # Use precomputed upper/lower lung masks for more accurate distribution
                lung_distribution = self._calculate_lung_distribution_with_masks(
                    lesion_array, lung_mask, upper_lung, lower_lung, spacing
                )
            else:
                # Fallback to standard calculation
                lung_distribution = self.lung_distribution_calc.calculate_lung_distribution(
                    lesion_array, lung_mask, spacing
                )
            
            lung_dist_summary = self.lung_distribution_calc.get_distribution_summary(
                lung_distribution
            )
            
            # Variable 3: Axial distribution (peripheral/central/diffuse)
            axial_distribution = self.axial_distribution_calc.calculate_axial_distribution(
                lesion_array, lung_mask, spacing
            )
            
            # Variable 4: Subpleural features
            subpleural_features = self.subpleural_calc.calculate_subpleural_features(
                lesion_array, lung_mask, spacing
            )
            subpleural_pattern = self.subpleural_calc.classify_subpleural_pattern(
                subpleural_features
            )
            
            # Compile all features
            all_features = {
                "lesion_type": lesion_features,
                "lung_distribution": lung_distribution,
                "axial_distribution": axial_distribution,
                "subpleural_features": subpleural_features
            }
            
            # Step 4: Pattern classification
            logger.info(f"Classifying pattern for {case_name}")
            pattern_classification = self.pattern_classifier.classify_pattern(all_features)
            classification_summary = self.pattern_classifier.get_classification_summary(
                pattern_classification
            )
            
            # Compile final results
            result.update({
                "success": True,
                "features": {
                    "lesion_type": {
                        "extraction": lesion_features,
                        "pattern": lesion_pattern,
                        "distribution": lesion_distribution
                    },
                    "lung_distribution": {
                        "analysis": lung_distribution,
                        "summary": lung_dist_summary
                    },
                    "axial_distribution": axial_distribution,
                    "subpleural_features": {
                        "analysis": subpleural_features,
                        "pattern": subpleural_pattern
                    }
                },
                "classification": {
                    "detailed": pattern_classification,
                    "summary": classification_summary
                },
                "four_variables_summary": self._generate_four_variables_summary(all_features),
                "lung_statistics": self.lung_segmentator.get_lung_statistics(lung_mask, spacing)
            })
            
            # Save intermediate results if requested
            if save_intermediate:
                self._save_intermediate_results(case_name, result, lung_mask, ct_sitk)
            
            logger.info(f"Successfully processed {case_name}: "
                       f"{classification_summary['pattern']} "
                       f"(confidence: {classification_summary['confidence']})")
            
        except Exception as e:
            error_msg = f"Error processing {case_name}: {str(e)}"
            logger.error(error_msg)
            result["error"] = error_msg
        
        return result
    
    def process_batch(self, case_list: List[str] = None, 
                     save_individual: bool = True) -> Dict[str, Any]:
        """
        Process multiple cases in batch
        
        Args:
            case_list: List of case names to process (default: all available)
            save_individual: Whether to save individual case results
            
        Returns:
            Dictionary with batch processing results
        """
        batch_result = {
            "batch_info": {
                "start_time": datetime.now().isoformat(),
                "total_cases": 0,
                "successful": 0,
                "failed": 0
            },
            "individual_results": [],
            "batch_summary": {},
            "failed_cases": []
        }
        
        try:
            # Get case list
            if case_list is None:
                case_list = self.data_loader.find_available_cases()
            
            batch_result["batch_info"]["total_cases"] = len(case_list)
            
            if not case_list:
                batch_result["error"] = "No cases found to process"
                return batch_result
            
            logger.info(f"Starting batch processing of {len(case_list)} cases")
            
            # Process each case
            successful_results = []
            
            for i, case_name in enumerate(case_list, 1):
                logger.info(f"Processing case {i}/{len(case_list)}: {case_name}")
                
                case_result = self.process_single_case(case_name, save_intermediate=False)
                batch_result["individual_results"].append(case_result)
                
                if case_result["success"]:
                    successful_results.append(case_result)
                    batch_result["batch_info"]["successful"] += 1
                    
                    # Save individual result
                    if save_individual:
                        self._save_individual_result(case_name, case_result)
                else:
                    batch_result["batch_info"]["failed"] += 1
                    batch_result["failed_cases"].append({
                        "case_name": case_name,
                        "error": case_result.get("error", "Unknown error")
                    })
            
            # Generate batch summary
            batch_result["batch_summary"] = self._generate_batch_summary(successful_results)
            batch_result["batch_info"]["end_time"] = datetime.now().isoformat()
            
            # Save batch results
            self._save_batch_results(batch_result)
            
            logger.info(f"Batch processing completed: "
                       f"{batch_result['batch_info']['successful']} successful, "
                       f"{batch_result['batch_info']['failed']} failed")
            
        except Exception as e:
            error_msg = f"Error in batch processing: {str(e)}"
            logger.error(error_msg)
            batch_result["error"] = error_msg
        
        return batch_result
    
    def _generate_four_variables_summary(self, features: Dict[str, Any]) -> Dict[str, str]:
        """Generate concise summary of the four variables"""
        summary = {
            "variable_1_lesion_type": "Unknown",
            "variable_2_lung_distribution": "Unknown", 
            "variable_3_axial_distribution": "Unknown",
            "variable_4_subpleural_features": "Unknown"
        }
        
        try:
            # Variable 1: Lesion type
            lesion_features = features.get("lesion_type", {})
            dominant_lesion = lesion_features.get("dominant_lesion", "none")
            lesion_types = lesion_features.get("lesion_types_present", [])
            
            if dominant_lesion and dominant_lesion != "none":
                summary["variable_1_lesion_type"] = f"Dominant: {dominant_lesion}"
            elif lesion_types:
                summary["variable_1_lesion_type"] = f"Mixed: {', '.join(lesion_types[:2])}"
            else:
                summary["variable_1_lesion_type"] = "No significant lesions"
            
            # Variable 2: Lung distribution
            lung_dist = features.get("lung_distribution", {})
            dist_pattern = lung_dist.get("distribution_pattern", "unknown")
            summary["variable_2_lung_distribution"] = dist_pattern.replace("_", " ").title()
            
            # Variable 3: Axial distribution
            axial_dist = features.get("axial_distribution", {})
            axial_pattern = axial_dist.get("axial_distribution", "unknown")
            summary["variable_3_axial_distribution"] = axial_pattern.replace("_", " ").title()
            
            # Variable 4: Subpleural features
            subpleural = features.get("subpleural_features", {})
            subpleural_involvement = subpleural.get("subpleural_involvement", False)
            if subpleural_involvement:
                ratio = subpleural.get("subpleural_lesion_ratio", 0) * 100
                summary["variable_4_subpleural_features"] = f"Present ({ratio:.1f}%)"
            else:
                summary["variable_4_subpleural_features"] = "Spared"
            
        except Exception as e:
            logger.error(f"Error generating four variables summary: {str(e)}")
            summary["error"] = str(e)
        
        return summary
    
    def _generate_batch_summary(self, successful_results: List[Dict]) -> Dict[str, Any]:
        """Generate summary statistics for batch processing"""
        summary = {
            "pattern_distribution": {},
            "feature_statistics": {},
            "performance_metrics": {}
        }
        
        try:
            if not successful_results:
                return summary
            
            # Pattern distribution
            patterns = [r["classification"]["summary"]["pattern"] for r in successful_results]
            pattern_counts = {}
            for pattern in patterns:
                pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
            
            summary["pattern_distribution"] = pattern_counts
            
            # Feature statistics
            lesion_types = []
            distributions = []
            axial_patterns = []
            subpleural_involvement = []
            
            for result in successful_results:
                features = result["features"]
                
                # Collect lesion types
                dominant = features["lesion_type"]["extraction"].get("dominant_lesion")
                if dominant:
                    lesion_types.append(dominant)
                
                # Collect distributions
                dist = features["lung_distribution"]["analysis"].get("distribution_pattern")
                if dist:
                    distributions.append(dist)
                
                # Collect axial patterns
                axial = features["axial_distribution"].get("axial_distribution")
                if axial:
                    axial_patterns.append(axial)
                
                # Collect subpleural involvement
                subpleural = features["subpleural_features"]["analysis"].get("subpleural_involvement", False)
                subpleural_involvement.append(subpleural)
            
            # Calculate statistics
            def count_items(items):
                counts = {}
                for item in items:
                    counts[item] = counts.get(item, 0) + 1
                return counts
            
            summary["feature_statistics"] = {
                "lesion_types": count_items(lesion_types),
                "lung_distributions": count_items(distributions),
                "axial_patterns": count_items(axial_patterns),
                "subpleural_involvement_rate": f"{sum(subpleural_involvement) / len(subpleural_involvement) * 100:.1f}%"
            }
            
        except Exception as e:
            logger.error(f"Error generating batch summary: {str(e)}")
            summary["error"] = str(e)
        
        return summary
    
    def _save_individual_result(self, case_name: str, result: Dict[str, Any]):
        """Save individual case result"""
        try:
            filename = f"{case_name}_analysis_result.json"
            filepath = os.path.join(self.output_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"Saved individual result: {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving individual result for {case_name}: {str(e)}")
    
    def _save_batch_results(self, batch_result: Dict[str, Any]):
        """Save batch processing results"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"batch_analysis_results_{timestamp}.json"
            filepath = os.path.join(self.output_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(batch_result, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved batch results: {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving batch results: {str(e)}")
    
    def _save_intermediate_results(self, case_name: str, result: Dict[str, Any], 
                                 lung_mask: np.ndarray, ct_sitk):
        """Save intermediate results including lung mask"""
        try:
            # Save lung mask
            import SimpleITK as sitk
            
            mask_sitk = sitk.GetImageFromArray(lung_mask)
            mask_sitk.SetSpacing(ct_sitk.GetSpacing())
            mask_sitk.SetOrigin(ct_sitk.GetOrigin())
            mask_sitk.SetDirection(ct_sitk.GetDirection())
            
            mask_filename = f"{case_name}_lung_mask.nii.gz"
            mask_filepath = os.path.join(self.output_dir, mask_filename)
            sitk.WriteImage(mask_sitk, mask_filepath)
            
            logger.debug(f"Saved lung mask: {mask_filepath}")
            
        except Exception as e:
            logger.error(f"Error saving intermediate results for {case_name}: {str(e)}")
    
    def validate_setup(self) -> Dict[str, Any]:
        """Validate pipeline setup and data availability"""
        validation = {
            "setup_valid": True,
            "directories": {},
            "available_cases": [],
            "issues": []
        }
        
        try:
            # Check directories
            validation["directories"]["ct_dir"] = {
                "path": self.ct_dir,
                "exists": os.path.exists(self.ct_dir)
            }
            validation["directories"]["lesion_dir"] = {
                "path": self.lesion_dir,
                "exists": os.path.exists(self.lesion_dir)
            }
            validation["directories"]["output_dir"] = {
                "path": self.output_dir,
                "exists": os.path.exists(self.output_dir)
            }
            
            # Check for issues
            if not validation["directories"]["ct_dir"]["exists"]:
                validation["issues"].append(f"CT directory not found: {self.ct_dir}")
                validation["setup_valid"] = False
            
            if not validation["directories"]["lesion_dir"]["exists"]:
                validation["issues"].append(f"Lesion directory not found: {self.lesion_dir}")
                validation["setup_valid"] = False
            
            # Find available cases
            if validation["setup_valid"]:
                validation["available_cases"] = self.data_loader.find_available_cases()
                
                if not validation["available_cases"]:
                    validation["issues"].append("No matching cases found")
                    validation["setup_valid"] = False
                else:
                    logger.info(f"Found {len(validation['available_cases'])} available cases")
            
        except Exception as e:
            validation["setup_valid"] = False
            validation["issues"].append(f"Validation error: {str(e)}")
        
        return validation
    
    def _calculate_lung_distribution_with_masks(self, lesion_mask: np.ndarray, 
                                              lung_mask: np.ndarray,
                                              upper_lung: np.ndarray, 
                                              lower_lung: np.ndarray,
                                              spacing: Tuple[float, float, float]) -> Dict[str, Any]:
        """
        Calculate lung distribution using precomputed upper/lower lung masks
        
        Args:
            lesion_mask: 3D lesion segmentation mask  
            lung_mask: 3D lung mask
            upper_lung: 3D upper lung mask
            lower_lung: 3D lower lung mask
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
            "upper_lesion_ratio": 0.0,
            "lower_lesion_ratio": 0.0,
            "upper_lesion_proportion": 0.0,
            "lower_lesion_proportion": 0.0,
            "distribution_pattern": "unknown",
            "z_cut_position": None,
            "upper_lung_voxels": 0,
            "lower_lung_voxels": 0,
            "upper_lesion_voxels": 0,
            "lower_lesion_voxels": 0
        }
        
        try:
            if spacing is None:
                spacing = (1.0, 1.0, 1.0)
            
            voxel_volume = spacing[0] * spacing[1] * spacing[2]
            
            # Calculate lung volumes
            upper_lung_voxels = np.sum(upper_lung > 0)
            lower_lung_voxels = np.sum(lower_lung > 0)
            total_lung_voxels = np.sum(lung_mask > 0)
            
            results["upper_lung_voxels"] = int(upper_lung_voxels)
            results["lower_lung_voxels"] = int(lower_lung_voxels)
            results["upper_lung_volume"] = upper_lung_voxels * voxel_volume
            results["lower_lung_volume"] = lower_lung_voxels * voxel_volume
            results["total_lung_volume"] = total_lung_voxels * voxel_volume
            
            # Calculate lesion volumes in each region
            upper_lesions = lesion_mask & upper_lung
            lower_lesions = lesion_mask & lower_lung
            total_lesions = lesion_mask & lung_mask
            
            upper_lesion_voxels = np.sum(upper_lesions > 0)
            lower_lesion_voxels = np.sum(lower_lesions > 0)
            total_lesion_voxels = np.sum(total_lesions > 0)
            
            results["upper_lesion_voxels"] = int(upper_lesion_voxels)
            results["lower_lesion_voxels"] = int(lower_lesion_voxels)
            results["upper_lesion_volume"] = upper_lesion_voxels * voxel_volume
            results["lower_lesion_volume"] = lower_lesion_voxels * voxel_volume
            results["total_lesion_volume"] = total_lesion_voxels * voxel_volume
            
            # Calculate ratios and proportions
            if upper_lung_voxels > 0:
                results["upper_lesion_ratio"] = upper_lesion_voxels / upper_lung_voxels
            
            if lower_lung_voxels > 0:
                results["lower_lesion_ratio"] = lower_lesion_voxels / lower_lung_voxels
            
            if total_lesion_voxels > 0:
                results["upper_lesion_proportion"] = upper_lesion_voxels / total_lesion_voxels
                results["lower_lesion_proportion"] = lower_lesion_voxels / total_lesion_voxels
            
            # Determine distribution pattern
            upper_prop = results["upper_lesion_proportion"]
            lower_prop = results["lower_lesion_proportion"]
            
            if upper_prop > 0.70:
                results["distribution_pattern"] = "upper_predominant"
            elif lower_prop > 0.70:
                results["distribution_pattern"] = "lower_predominant"
            elif abs(upper_prop - lower_prop) <= 0.20:
                results["distribution_pattern"] = "diffuse"
            else:
                results["distribution_pattern"] = "mixed"
            
            # Estimate z_cut position (approximate)
            upper_z = np.where(upper_lung)[0]
            lower_z = np.where(lower_lung)[0]
            
            if len(upper_z) > 0 and len(lower_z) > 0:
                results["z_cut_position"] = int(np.mean([np.min(upper_z), np.max(lower_z)]))
            
            logger.debug(f"Calculated lung distribution with precomputed masks: "
                        f"{results['distribution_pattern']} "
                        f"(upper: {upper_prop:.3f}, lower: {lower_prop:.3f})")
            
        except Exception as e:
            logger.error(f"Error calculating lung distribution with precomputed masks: {str(e)}")
            
        return results