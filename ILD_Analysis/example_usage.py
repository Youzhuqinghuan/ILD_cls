"""
Example usage script for ILD Analysis Pipeline
Demonstrates how to use the pipeline programmatically
"""

import os
import sys
import logging

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pipeline.full_pipeline import ILDAnalysisPipeline
from config import Config

def setup_logging():
    """Setup basic logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def example_single_case():
    """Example: Process a single case"""
    print("="*60)
    print("EXAMPLE: Processing Single Case")
    print("="*60)
    
    # Initialize pipeline
    pipeline = ILDAnalysisPipeline()
    
    # Validate setup first
    validation = pipeline.validate_setup()
    if not validation["setup_valid"]:
        print("Pipeline setup is invalid:")
        for issue in validation["issues"]:
            print(f"  - {issue}")
        return
    
    # Get first available case
    available_cases = validation["available_cases"]
    if not available_cases:
        print("No cases available for processing")
        return
    
    case_name = available_cases[0]
    print(f"Processing case: {case_name}")
    
    # Process the case
    result = pipeline.process_single_case(case_name, save_intermediate=True)
    
    if result["success"]:
        print("✓ Processing successful!")
        
        # Print key results
        classification = result["classification"]["summary"]
        four_vars = result["four_variables_summary"]
        
        print(f"\nResults:")
        print(f"  Predicted Pattern: {classification['pattern']}")
        print(f"  Confidence: {classification['confidence']}")
        print(f"  Certainty: {classification['certainty']}")
        
        print(f"\nFour Variables:")
        print(f"  1. Lesion Type: {four_vars['variable_1_lesion_type']}")
        print(f"  2. Lung Distribution: {four_vars['variable_2_lung_distribution']}")
        print(f"  3. Axial Distribution: {four_vars['variable_3_axial_distribution']}")
        print(f"  4. Subpleural Features: {four_vars['variable_4_subpleural_features']}")
        
        # Print detailed classification reasoning
        detailed = result["classification"]["detailed"]
        if "reasoning" in detailed and detailed["reasoning"]:
            print(f"\nClassification Reasoning:")
            for reason in detailed["reasoning"][:5]:  # Show first 5 reasons
                print(f"  - {reason}")
    else:
        print(f"✗ Processing failed: {result.get('error', 'Unknown error')}")

def example_batch_processing():
    """Example: Process multiple cases"""
    print("\n" + "="*60)
    print("EXAMPLE: Batch Processing")
    print("="*60)
    
    # Initialize pipeline
    pipeline = ILDAnalysisPipeline()
    
    # Get available cases
    available_cases = pipeline.data_loader.find_available_cases()
    
    if len(available_cases) < 2:
        print("Need at least 2 cases for batch processing example")
        return
    
    # Process first 3 cases (or all if less than 3)
    case_subset = available_cases[:min(3, len(available_cases))]
    print(f"Processing cases: {', '.join(case_subset)}")
    
    # Run batch processing
    batch_result = pipeline.process_batch(case_list=case_subset, save_individual=True)
    
    # Print results
    batch_info = batch_result["batch_info"]
    print(f"\nBatch Results:")
    print(f"  Total: {batch_info['total_cases']}")
    print(f"  Successful: {batch_info['successful']}")
    print(f"  Failed: {batch_info['failed']}")
    
    # Print individual results
    if batch_result["individual_results"]:
        print(f"\nIndividual Results:")
        for result in batch_result["individual_results"]:
            case_name = result["case_name"]
            if result["success"]:
                pattern = result["classification"]["summary"]["pattern"]
                confidence = result["classification"]["summary"]["confidence"]
                print(f"  ✓ {case_name}: {pattern} ({confidence})")
            else:
                error = result.get("error", "Unknown error")
                print(f"  ✗ {case_name}: {error}")
    
    # Print pattern distribution
    if "batch_summary" in batch_result and "pattern_distribution" in batch_result["batch_summary"]:
        pattern_dist = batch_result["batch_summary"]["pattern_distribution"]
        print(f"\nPattern Distribution:")
        for pattern, count in pattern_dist.items():
            print(f"  {pattern}: {count}")

def example_feature_extraction():
    """Example: Extract features without classification"""
    print("\n" + "="*60)
    print("EXAMPLE: Feature Extraction Only")
    print("="*60)
    
    # Initialize individual components
    from core.data_loader import ILDDataLoader
    from core.lung_segmentation import LungSegmentator
    from features.lesion_type import LesionTypeExtractor
    
    # Setup
    data_loader = ILDDataLoader(Config.CT_IMAGES_DIR, Config.PREDICTED_LABELS_DIR)
    lung_segmentator = LungSegmentator()
    lesion_extractor = LesionTypeExtractor()
    
    # Get first available case
    available_cases = data_loader.find_available_cases()
    if not available_cases:
        print("No cases available")
        return
    
    case_name = available_cases[0]
    print(f"Extracting features for: {case_name}")
    
    # Load data
    ct_array, lesion_array, spacing, ct_sitk = data_loader.load_case(case_name)
    
    if ct_array is None:
        print("Failed to load case data")
        return
    
    # Segment lungs
    lung_mask = lung_segmentator.segment_volume(ct_array, spacing)
    
    # Extract lesion features
    lesion_features = lesion_extractor.extract_lesion_types(lesion_array, spacing)
    
    print(f"\nLesion Features:")
    print(f"  Lesion types present: {lesion_features.get('lesion_types_present', [])}")
    print(f"  Dominant lesion: {lesion_features.get('dominant_lesion', 'None')}")
    print(f"  Total lesion volume: {lesion_features.get('total_lesion_volume', 0):.2f} mm³")
    
    # Get lung statistics
    lung_stats = lung_segmentator.get_lung_statistics(lung_mask, spacing)
    print(f"\nLung Statistics:")
    print(f"  Total lung volume: {lung_stats.get('total_volume_mm3', 0):.2f} mm³")
    print(f"  Upper lung volume: {lung_stats.get('upper_volume_mm3', 0):.2f} mm³")
    print(f"  Lower lung volume: {lung_stats.get('lower_volume_mm3', 0):.2f} mm³")

def main():
    """Run all examples"""
    setup_logging()
    
    print("ILD Analysis Pipeline - Usage Examples")
    print("=====================================")
    
    try:
        # Run examples
        example_single_case()
        example_batch_processing()
        example_feature_extraction()
        
        print("\n" + "="*60)
        print("All examples completed!")
        print("Check the output directory for saved results.")
        print("="*60)
        
    except Exception as e:
        print(f"Error running examples: {str(e)}")
        logging.error(f"Error in examples: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()