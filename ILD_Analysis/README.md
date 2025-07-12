# ILD Analysis Pipeline

A modular system for analyzing Interstitial Lung Disease (ILD) patterns from HRCT images. The pipeline extracts four key variables and classifies cases into UIP, NSIP, OP, Other, or Normal patterns.

## Overview

This pipeline implements a comprehensive workflow:

1. **Data Loading**: Reads CT images and ILD_Segmentation predictions
2. **Lung Segmentation**: Unified lung segmentation with upper/lower division  
3. **Feature Extraction**: Calculates four key variables:
   - Variable 1: Lesion type (from predicted labels)
   - Variable 2: Lung distribution (upper/lower predominance)
   - Variable 3: Axial distribution (peripheral/central/diffuse)
   - Variable 4: Subpleural features (involvement analysis)
4. **Pattern Classification**: Rule-based classification into ILD patterns

## Architecture

```
ILD_Analysis/
├── core/                    # Core utilities
│   ├── data_loader.py      # Unified data loading
│   └── lung_segmentation.py # Lung segmentation
├── features/               # Feature extractors
│   ├── lesion_type.py     # Variable 1: Lesion types
│   ├── lung_distribution.py # Variable 2: Upper/lower distribution
│   ├── axial_distribution.py # Variable 3: Peripheral/central
│   └── subpleural_features.py # Variable 4: Subpleural analysis
├── classification/         # Pattern classification
│   └── pattern_classifier.py # UIP/NSIP/OP classifier
├── pipeline/              # Pipeline orchestration
│   └── full_pipeline.py  # Complete workflow
├── config.py             # Configuration settings
├── main.py              # Command-line interface
└── example_usage.py     # Usage examples
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure data directories exist:
   - CT images: `dataset/images/`
   - Lesion predictions: `ILD_Segmentation/segs/`

## Usage

### Command Line Interface

**Validate setup:**
```bash
python main.py --validate
```

**Process single case:**
```bash
python main.py --case NSIP1 --save-intermediate
```

**Batch processing:**
```bash
python main.py --batch
```

**Custom directories:**
```bash
python main.py --ct-dir /path/to/ct --lesion-dir /path/to/lesions --batch
```

### Programmatic Usage

```python
from pipeline.full_pipeline import ILDAnalysisPipeline

# Initialize pipeline
pipeline = ILDAnalysisPipeline()

# Process single case
result = pipeline.process_single_case("NSIP1")

# Process batch
batch_result = pipeline.process_batch()
```

## Data Requirements

### Input Files
- **CT Images**: `CT_{case_name}_0000.nii.gz` (in dataset/images/)
- **Lesion Predictions**: `CT_{case_name}.nii.gz` (in ILD_Segmentation/segs/)

### Lesion Labels
- 0: Background
- 1: Honeycomb
- 2: Reticulation  
- 3: Ground glass opacity (GGO)
- 4: Consolidation

## Output

### Individual Case Results
Each case generates a JSON file with:
- **Four Variables Summary**: Concise summary of extracted variables
- **Feature Details**: Complete feature extraction results
- **Classification**: Pattern prediction with confidence and reasoning
- **Lung Statistics**: Volume measurements and statistics

### Batch Results
Batch processing generates:
- **Individual Results**: Separate JSON file per case
- **Batch Summary**: Aggregate statistics and pattern distribution
- **Failed Cases**: List of cases that failed processing

## Four Variables

### Variable 1: Lesion Type
- Analyzes predicted lesion labels
- Identifies dominant lesion type
- Classifies lesion patterns (UIP-like, NSIP-like, OP-like)

### Variable 2: Lung Distribution  
- Divides lungs into upper/lower regions (60th percentile)
- Calculates lesion distribution between regions
- Classifies as upper-predominant, lower-predominant, or diffuse

### Variable 3: Axial Distribution
- Calculates distance from pleural surface
- Uses 10mm + 20% relative thresholds for peripheral classification
- Classifies as peripheral, central, or diffuse/scattered

### Variable 4: Subpleural Features
- Analyzes lesions within 3mm of pleural surface
- Calculates subpleural involvement ratio
- Classifies subpleural pattern and severity

## Classification Rules

### UIP Pattern
- Honeycomb pattern present (strong indicator)
- Lower lobe predominance
- Peripheral distribution
- Subpleural involvement

### NSIP Pattern  
- Ground glass ± reticulation
- No honeycomb pattern
- Subpleural sparing (classic feature)
- Upper or diffuse distribution

### OP Pattern
- Consolidation ± ground glass
- No fibrotic patterns (honeycomb/reticulation)
- Variable distribution
- Peripheral or subpleural involvement

## Configuration

Key parameters in `config.py`:
- `LUNG_THRESHOLD = -300` # HU threshold for lung segmentation
- `UPPER_LOWER_PERCENTILE = 60` # Upper/lower lung division
- `PERIPHERAL_THRESHOLD_MM = 10` # Peripheral classification threshold
- `SUBPLEURAL_THRESHOLD_MM = 3.0` # Subpleural involvement threshold

## Examples

See `example_usage.py` for comprehensive examples:
- Single case processing
- Batch processing
- Feature extraction only
- Programmatic usage

## Error Handling

The pipeline includes comprehensive error handling:
- Data validation before processing
- Graceful failure with detailed error messages
- Continuation of batch processing despite individual failures
- Logging at multiple levels (DEBUG, INFO, WARNING, ERROR)

## Integration with ILD_Segmentation

The pipeline reads lesion predictions from `ILD_Segmentation/segs/` and expects:
- 4-class segmentation masks (background, honeycomb, reticulation, GGO, consolidation)
- Same spatial dimensions as input CT images
- Standard NIfTI format (.nii.gz)

This modular design allows for easy integration with the existing ILD_Segmentation workflow while eliminating code redundancy and providing a unified analysis pipeline.