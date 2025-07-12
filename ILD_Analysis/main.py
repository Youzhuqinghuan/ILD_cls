"""
Main entry point for ILD Analysis
Provides command-line interface for running the complete analysis pipeline
"""

import os
import sys
import argparse
import logging
import json
from typing import List, Optional

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pipeline.full_pipeline import ILDAnalysisPipeline
from config import Config

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """Setup logging configuration"""
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Setup handlers
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    # Configure logging
    logging.basicConfig(
        level=level,
        handlers=handlers,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Set formatter for all handlers
    for handler in logging.getLogger().handlers:
        handler.setFormatter(formatter)

def main():
    """Main function with command-line interface"""
    parser = argparse.ArgumentParser(
        description='ILD Analysis Pipeline - Complete analysis from CT images and lesion predictions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single case
  python main.py --case NSIP1 --save-intermediate
  
  # Process all available cases
  python main.py --batch
  
  # Process specific cases
  python main.py --batch --cases NSIP1,NSIP2,UIP1
  
  # Validate setup
  python main.py --validate
  
  # Use custom directories
  python main.py --ct-dir /path/to/ct --lesion-dir /path/to/lesions --batch
        """
    )
    
    # Data directories
    parser.add_argument('--ct-dir', type=str, default=Config.CT_IMAGES_DIR,
                       help=f'CT images directory (default: {Config.CT_IMAGES_DIR})')
    parser.add_argument('--lesion-dir', type=str, default=Config.PREDICTED_LABELS_DIR,
                       help=f'Lesion predictions directory (default: {Config.PREDICTED_LABELS_DIR})')
    parser.add_argument('--output-dir', type=str, default=Config.OUTPUT_DIR,
                       help=f'Output directory (default: {Config.OUTPUT_DIR})')
    
    # Processing options
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--case', type=str,
                      help='Process single case (e.g., NSIP1)')
    group.add_argument('--batch', action='store_true',
                      help='Process all available cases')
    group.add_argument('--validate', action='store_true',
                      help='Validate setup and show available cases')
    
    # Additional options
    parser.add_argument('--cases', type=str,
                       help='Comma-separated list of cases for batch processing')
    parser.add_argument('--save-intermediate', action='store_true',
                       help='Save intermediate results (lung masks, etc.)')
    parser.add_argument('--save-individual', action='store_true', default=True,
                       help='Save individual case results (default: True)')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level (default: INFO)')
    parser.add_argument('--log-file', type=str,
                       help='Log file path (default: console only)')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize pipeline
        logger.info("Initializing ILD Analysis Pipeline...")
        pipeline = ILDAnalysisPipeline(
            ct_dir=args.ct_dir,
            lesion_dir=args.lesion_dir,
            output_dir=args.output_dir
        )
        
        # Validate setup
        if args.validate:
            logger.info("Validating pipeline setup...")
            validation = pipeline.validate_setup()
            
            print("\n" + "="*60)
            print("PIPELINE VALIDATION RESULTS")
            print("="*60)
            
            # Print directory status
            print("\nDirectories:")
            for dir_name, dir_info in validation["directories"].items():
                status = "✓" if dir_info["exists"] else "✗"
                print(f"  {status} {dir_name}: {dir_info['path']}")
            
            # Print available cases
            if validation["available_cases"]:
                print(f"\nAvailable cases ({len(validation['available_cases'])}):")
                for i, case in enumerate(validation["available_cases"][:10], 1):
                    print(f"  {i:2d}. {case}")
                if len(validation["available_cases"]) > 10:
                    print(f"  ... and {len(validation['available_cases']) - 10} more")
            
            # Print issues
            if validation["issues"]:
                print("\nIssues found:")
                for issue in validation["issues"]:
                    print(f"  ✗ {issue}")
            
            # Print status
            status = "VALID" if validation["setup_valid"] else "INVALID"
            print(f"\nSetup Status: {status}")
            print("="*60)
            
            if not validation["setup_valid"]:
                sys.exit(1)
            
            return
        
        # Process single case
        if args.case:
            logger.info(f"Processing single case: {args.case}")
            
            result = pipeline.process_single_case(
                args.case, 
                save_intermediate=args.save_intermediate
            )
            
            if result["success"]:
                print(f"\n✓ Successfully processed case: {args.case}")
                
                # Print summary
                summary = result["classification"]["summary"]
                print(f"  Pattern: {summary['pattern']}")
                print(f"  Confidence: {summary['confidence']}")
                print(f"  Certainty: {summary['certainty']}")
                
                # Print four variables
                four_vars = result["four_variables_summary"]
                print(f"\n  Four Variables:")
                print(f"    1. Lesion Type: {four_vars['variable_1_lesion_type']}")
                print(f"    2. Lung Distribution: {four_vars['variable_2_lung_distribution']}")
                print(f"    3. Axial Distribution: {four_vars['variable_3_axial_distribution']}")
                print(f"    4. Subpleural Features: {four_vars['variable_4_subpleural_features']}")
                
                # Save result
                if args.save_individual:
                    pipeline._save_individual_result(args.case, result)
                    print(f"  Results saved to: {args.output_dir}")
                
            else:
                print(f"\n✗ Failed to process case: {args.case}")
                print(f"  Error: {result.get('error', 'Unknown error')}")
                sys.exit(1)
        
        # Process batch
        elif args.batch:
            logger.info("Starting batch processing...")
            
            # Parse case list if provided
            case_list = None
            if args.cases:
                case_list = [case.strip() for case in args.cases.split(',')]
                logger.info(f"Processing specified cases: {case_list}")
            
            batch_result = pipeline.process_batch(
                case_list=case_list,
                save_individual=args.save_individual
            )
            
            # Print batch summary
            batch_info = batch_result["batch_info"]
            print(f"\n{'='*60}")
            print("BATCH PROCESSING RESULTS")
            print(f"{'='*60}")
            print(f"Total cases: {batch_info['total_cases']}")
            print(f"Successful: {batch_info['successful']}")
            print(f"Failed: {batch_info['failed']}")
            
            if batch_result["batch_summary"]:
                summary = batch_result["batch_summary"]
                
                # Pattern distribution
                if "pattern_distribution" in summary:
                    print(f"\nPattern Distribution:")
                    for pattern, count in summary["pattern_distribution"].items():
                        print(f"  {pattern}: {count}")
                
                # Feature statistics
                if "feature_statistics" in summary:
                    stats = summary["feature_statistics"]
                    print(f"\nFeature Statistics:")
                    
                    if "subpleural_involvement_rate" in stats:
                        print(f"  Subpleural involvement: {stats['subpleural_involvement_rate']}")
                    
                    if "lesion_types" in stats:
                        print(f"  Dominant lesion types:")
                        for lesion_type, count in stats["lesion_types"].items():
                            print(f"    {lesion_type}: {count}")
            
            # Failed cases
            if batch_result["failed_cases"]:
                print(f"\nFailed Cases:")
                for failed in batch_result["failed_cases"]:
                    print(f"  ✗ {failed['case_name']}: {failed['error']}")
            
            print(f"\nResults saved to: {args.output_dir}")
            print(f"{'='*60}")
    
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()