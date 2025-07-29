#!/usr/bin/env python3
"""
Lung Mask Generation Script for ILD Analysis
Preprocesses all CT images to generate high-quality lung masks using lungmask
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any
import json
from datetime import datetime
import traceback

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

# Import our custom modules
from core.lungmask_segmentation import LungmaskSegmentator
from config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LungMaskGenerator:
    """Batch lung mask generation using lungmask"""
    
    def __init__(self, ct_dir: str, output_dir: str, enable_lobes: bool = False):
        """
        Initialize lung mask generator
        
        Args:
            ct_dir: Directory containing CT images
            output_dir: Directory to save lung masks
            enable_lobes: Whether to enable lobe-based segmentation
        """
        self.ct_dir = Path(ct_dir)
        self.output_dir = Path(output_dir)
        self.enable_lobes = enable_lobes
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize segmentator
        try:
            self.segmentator = LungmaskSegmentator(enable_lobe_segmentation=enable_lobes)
            logger.info(f"Initialized lungmask segmentator (lobes: {enable_lobes})")
        except Exception as e:
            logger.error(f"Failed to initialize lungmask: {str(e)}")
            raise
    
    def find_ct_files(self) -> List[Dict[str, str]]:
        """
        Find all CT files in the input directory
        
        Returns:
            List of dictionaries with file info
        """
        ct_files = []
        
        # Pattern: CT_<case_name>_0000.nii.gz
        for ct_file in self.ct_dir.glob("CT_*_0000.nii.gz"):
            filename = ct_file.name
            # Extract case name: CT_NSIP1_0000.nii.gz -> NSIP1
            parts = filename.replace(".nii.gz", "").split("_")
            if len(parts) >= 3:
                case_name = "_".join(parts[1:-1])  # Handle cases like "NSIP 1" -> "NSIP 1"
                
                ct_files.append({
                    'case_name': case_name,
                    'filename': filename,
                    'full_path': str(ct_file)
                })
        
        logger.info(f"Found {len(ct_files)} CT files")
        return sorted(ct_files, key=lambda x: x['case_name'])
    
    def load_ct_image(self, ct_path: str) -> tuple:
        """
        Load CT image and extract metadata
        
        Args:
            ct_path: Path to CT file
            
        Returns:
            Tuple of (ct_array, spacing, sitk_image)
        """
        try:
            # Load with SimpleITK
            sitk_image = sitk.ReadImage(ct_path)
            
            # Convert to numpy array
            ct_array = sitk.GetArrayFromImage(sitk_image)
            
            # Get spacing (convert from ITK (x,y,z) to (z,y,x))
            spacing = sitk_image.GetSpacing()[::-1]
            
            logger.debug(f"Loaded CT: shape={ct_array.shape}, spacing={spacing}")
            return ct_array, spacing, sitk_image
            
        except Exception as e:
            logger.error(f"Error loading CT image {ct_path}: {str(e)}")
            return None, None, None
    
    def process_single_case(self, case_info: Dict[str, str]) -> Dict[str, Any]:
        """
        Process a single case
        
        Args:
            case_info: Dictionary with case information
            
        Returns:
            Result dictionary
        """
        case_name = case_info['case_name']
        ct_path = case_info['full_path']
        
        result = {
            'case_name': case_name,
            'success': False,
            'timestamp': datetime.now().isoformat(),
            'files_generated': [],
            'statistics': {},
            'error': None
        }
        
        try:
            logger.info(f"Processing case: {case_name}")
            
            # Check if outputs already exist
            expected_files = [
                self.output_dir / f"CT_{case_name}_lung_mask.nii.gz",
                self.output_dir / f"CT_{case_name}_upper_lung.nii.gz", 
                self.output_dir / f"CT_{case_name}_lower_lung.nii.gz"
            ]
            
            if self.enable_lobes:
                expected_files.append(self.output_dir / f"CT_{case_name}_lobe_mask.nii.gz")
            
            # Check if all files exist
            if all(f.exists() for f in expected_files):
                logger.info(f"Lung masks already exist for {case_name}, skipping")
                result.update({
                    'success': True,
                    'files_generated': [str(f) for f in expected_files],
                    'skipped': True
                })
                return result
            
            # Load CT image
            ct_array, spacing, sitk_image = self.load_ct_image(ct_path)
            
            if ct_array is None:
                result['error'] = "Failed to load CT image"
                return result
            
            result['input_info'] = {
                'ct_shape': ct_array.shape,
                'spacing': spacing,
                'file_size_mb': os.path.getsize(ct_path) / (1024*1024)
            }
            
            # Perform segmentation and save
            logger.info(f"Generating lung masks for {case_name}...")
            
            start_time = datetime.now()
            
            segmentation_result = self.segmentator.precompute_and_save(
                ct_array, spacing, case_name, str(self.output_dir)
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            if segmentation_result['success']:
                result.update({
                    'success': True,
                    'files_generated': list(segmentation_result['files'].values()),
                    'statistics': segmentation_result['statistics'],
                    'processing_time_seconds': processing_time
                })
                
                logger.info(f"Successfully processed {case_name} in {processing_time:.1f}s: "
                           f"{len(segmentation_result['files'])} files generated")
            else:
                result['error'] = segmentation_result.get('error', 'Unknown segmentation error')
                
        except Exception as e:
            error_msg = f"Error processing {case_name}: {str(e)}"
            logger.error(error_msg)
            logger.debug(traceback.format_exc())
            result['error'] = error_msg
        
        return result
    
    def process_batch(self, case_list: List[str] = None, 
                     max_workers: int = 1) -> Dict[str, Any]:
        """
        Process multiple cases in batch
        
        Args:
            case_list: List of specific cases to process (default: all)
            max_workers: Number of parallel workers (currently supports only 1)
            
        Returns:
            Batch processing results
        """
        batch_result = {
            'batch_info': {
                'start_time': datetime.now().isoformat(),
                'total_cases': 0,
                'successful': 0,
                'failed': 0,
                'skipped': 0
            },
            'individual_results': [],
            'failed_cases': [],
            'summary': {}
        }
        
        try:
            # Find all CT files
            ct_files = self.find_ct_files()
            
            # Filter by case list if provided
            if case_list:
                ct_files = [f for f in ct_files if f['case_name'] in case_list]
            
            batch_result['batch_info']['total_cases'] = len(ct_files)
            
            if not ct_files:
                batch_result['error'] = "No CT files found to process"
                return batch_result
            
            logger.info(f"Starting batch processing of {len(ct_files)} cases")
            
            # Process each case with progress bar
            successful_results = []
            
            for ct_file in tqdm(ct_files, desc="Processing cases"):
                case_result = self.process_single_case(ct_file)
                batch_result['individual_results'].append(case_result)
                
                if case_result['success']:
                    successful_results.append(case_result)
                    batch_result['batch_info']['successful'] += 1
                    
                    # Check if skipped
                    if case_result.get('skipped', False):
                        batch_result['batch_info']['skipped'] += 1
                else:
                    batch_result['batch_info']['failed'] += 1
                    batch_result['failed_cases'].append({
                        'case_name': case_result['case_name'],
                        'error': case_result.get('error', 'Unknown error')
                    })
            
            # Generate summary
            batch_result['summary'] = self._generate_batch_summary(successful_results)
            batch_result['batch_info']['end_time'] = datetime.now().isoformat()
            
            # Save batch results
            self._save_batch_results(batch_result)
            
            logger.info(f"Batch processing completed: "
                       f"{batch_result['batch_info']['successful']} successful, "
                       f"{batch_result['batch_info']['skipped']} skipped, "
                       f"{batch_result['batch_info']['failed']} failed")
            
        except Exception as e:
            error_msg = f"Error in batch processing: {str(e)}"
            logger.error(error_msg)
            batch_result['error'] = error_msg
        
        return batch_result
    
    def _generate_batch_summary(self, successful_results: List[Dict]) -> Dict[str, Any]:
        """Generate summary statistics"""
        if not successful_results:
            return {}
        
        # Filter out skipped results for statistics
        processed_results = [r for r in successful_results if not r.get('skipped', False)]
        
        if not processed_results:
            return {'note': 'All cases were skipped (already processed)'}
        
        # Calculate processing time statistics
        processing_times = [r.get('processing_time_seconds', 0) for r in processed_results]
        
        # Calculate volume statistics
        volumes = []
        for result in processed_results:
            stats = result.get('statistics', {})
            volume = stats.get('total_volume_mm3', 0)
            if volume > 0:
                volumes.append(volume)
        
        summary = {
            'processing_statistics': {
                'processed_cases': len(processed_results),
                'avg_processing_time_seconds': np.mean(processing_times) if processing_times else 0,
                'total_processing_time_seconds': sum(processing_times)
            }
        }
        
        if volumes:
            summary['volume_statistics'] = {
                'avg_lung_volume_mm3': np.mean(volumes),
                'min_lung_volume_mm3': np.min(volumes),
                'max_lung_volume_mm3': np.max(volumes),
                'std_lung_volume_mm3': np.std(volumes)
            }
        
        return summary
    
    def _save_batch_results(self, batch_result: Dict[str, Any]):
        """Save batch processing results"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"lung_mask_generation_results_{timestamp}.json"
            filepath = self.output_dir / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(batch_result, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Saved batch results: {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving batch results: {str(e)}")
    
    def validate_setup(self) -> Dict[str, Any]:
        """Validate setup and dependencies"""
        validation = {
            'setup_valid': True,
            'issues': [],
            'dependencies': {},
            'directories': {}
        }
        
        try:
            # Check directories
            validation['directories']['ct_dir'] = {
                'path': str(self.ct_dir),
                'exists': self.ct_dir.exists(),
                'files_count': len(list(self.ct_dir.glob("CT_*_0000.nii.gz")))
            }
            
            validation['directories']['output_dir'] = {
                'path': str(self.output_dir),
                'exists': self.output_dir.exists(),
                'writable': os.access(str(self.output_dir), os.W_OK)
            }
            
            # Check dependencies
            try:
                import lungmask
                validation['dependencies']['lungmask'] = {'available': True, 'version': getattr(lungmask, '__version__', 'unknown')}
            except ImportError:
                validation['dependencies']['lungmask'] = {'available': False}
                validation['issues'].append("lungmask package not available")
                validation['setup_valid'] = False
            
            try:
                import SimpleITK as sitk
                validation['dependencies']['SimpleITK'] = {'available': True, 'version': sitk.Version.VersionString()}
            except ImportError:
                validation['dependencies']['SimpleITK'] = {'available': False}
                validation['issues'].append("SimpleITK package not available")
                validation['setup_valid'] = False
            
            # Validate CT directory
            if not validation['directories']['ct_dir']['exists']:
                validation['issues'].append(f"CT directory not found: {self.ct_dir}")
                validation['setup_valid'] = False
            elif validation['directories']['ct_dir']['files_count'] == 0:
                validation['issues'].append("No CT files found (pattern: CT_*_0000.nii.gz)")
                validation['setup_valid'] = False
            
            # Validate output directory
            if not validation['directories']['output_dir']['writable']:
                validation['issues'].append(f"Output directory not writable: {self.output_dir}")
                validation['setup_valid'] = False
            
        except Exception as e:
            validation['setup_valid'] = False
            validation['issues'].append(f"Validation error: {str(e)}")
        
        return validation


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Generate lung masks for ILD analysis using lungmask",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all cases
  python generate_lung_masks.py
  
  # Process specific cases
  python generate_lung_masks.py --cases NSIP1,UIP2,OP3
  
  # Validate setup only
  python generate_lung_masks.py --validate-only
  
  # Custom directories
  python generate_lung_masks.py --ct-dir /path/to/images --output-dir /path/to/lungs
        """
    )
    
    parser.add_argument('--ct-dir', type=str, default=None,
                       help='Directory containing CT images (default from config)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Directory to save lung masks (default: dataset/lungs)')
    parser.add_argument('--cases', type=str, default=None,
                       help='Comma-separated list of cases to process (default: all)')
    parser.add_argument('--enable-lobes', action='store_true',
                       help='Enable lobe-based segmentation (slower but more accurate upper/lower division)')
    parser.add_argument('--validate-only', action='store_true',
                       help='Only validate setup without processing')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help='Logging level')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Set directories
    ct_dir = args.ct_dir or Config.CT_IMAGES_DIR
    output_dir = args.output_dir or os.path.join(os.path.dirname(Config.CT_IMAGES_DIR), 'lungs')
    
    try:
        # Initialize generator
        generator = LungMaskGenerator(
            ct_dir=ct_dir,
            output_dir=output_dir,
            enable_lobes=args.enable_lobes
        )
        
        # Validate setup
        validation = generator.validate_setup()
        
        if not validation['setup_valid']:
            logger.error("Setup validation failed:")
            for issue in validation['issues']:
                logger.error(f"  - {issue}")
            sys.exit(1)
        
        logger.info("Setup validation passed")
        
        if args.validate_only:
            logger.info("Validation complete. Use --help for processing options.")
            return
        
        # Parse case list
        case_list = None
        if args.cases:
            case_list = [case.strip() for case in args.cases.split(',')]
            logger.info(f"Processing specific cases: {case_list}")
        
        # Process cases
        result = generator.process_batch(case_list)
        
        # Print summary
        if result.get('batch_info'):
            info = result['batch_info']
            logger.info(f"Batch processing summary:")
            logger.info(f"  Total cases: {info['total_cases']}")
            logger.info(f"  Successful: {info['successful']}")
            logger.info(f"  Skipped: {info['skipped']}")
            logger.info(f"  Failed: {info['failed']}")
            
            if info['failed'] > 0:
                logger.warning("Failed cases:")
                for failed in result.get('failed_cases', []):
                    logger.warning(f"  - {failed['case_name']}: {failed['error']}")
        
        # Exit with error code if any failures
        if result.get('batch_info', {}).get('failed', 0) > 0:
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.debug(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()