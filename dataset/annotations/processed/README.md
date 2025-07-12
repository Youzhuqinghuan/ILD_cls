# Medical Image Annotation Data Processing Results

## Overview

This directory contains processed results from the original CSV annotation file (`labels.csv`). The data has been simplified and converted to JSONL format for easy model consumption.

## Processing Logic

1. **File Grouping**: Data grouped by "图像清洗后编码" (Image ID after cleaning)
2. **Disease Pattern Extraction**: Extract disease type from filename (UIP, NSIP, AHP, OP, etc.)
3. **Statistical Analysis**: Count occurrences of axial distributions and manifestations for each file
4. **Feature Extraction**: Select most frequent axial distribution and manifestation as overall file features
5. **Translation**: Convert Chinese terms to English equivalents
6. **ROI Information**: Preserve detailed information for each ROI layer as lists

## Output Format

### processed_labels.jsonl
**Format**: JSONL (JSON Lines)
**Usage**: One JSON object per line, suitable for streaming and model training

**Data Structure**:
```json
{
  "filename": "file_identifier",
  "disease_pattern": "disease_type",
  "overall_axial_distribution": "most_frequent_axial_distribution",
  "overall_manifestation": "most_frequent_manifestation",
  "abnormal_manifestation_presence": [int, int, int, int],
  "roi_layers": [layer_numbers],
  "roi_axial_distributions": ["axial_distribution_per_roi"],
  "roi_manifestations": ["manifestation_per_roi"]
}
```

### Abnormal Manifestation Presence Logic
The `abnormal_manifestation_presence` field is a 4-element boolean list (represented as 0s and 1s) indicating the presence of specific abnormal manifestations in each file:
- **Index 0**: Honeycombing (蜂窝) - 1 if present, 0 if absent
- **Index 1**: Reticulation (网格) - 1 if present, 0 if absent  
- **Index 2**: Consolidation (实变) - 1 if present, 0 if absent
- **Index 3**: Ground-Glass Opacity (GGO) - 1 if present, 0 if absent

This boolean representation allows for easy identification of which abnormal manifestations are present in each medical image file.

**Note**: The global statistics for abnormal manifestations are counted at the ROI level, meaning each occurrence of an abnormal manifestation in a specific ROI is counted separately.
```

## Data Statistics

### Overall Statistics
- **Total Files**: 96
- **Total ROIs**: 2121
- **Average ROIs per File**: 22.09
- **Min ROIs per File**: 11
- **Max ROIs per File**: 52

### Abnormal Manifestation Statistics (ROI Level)
- **consolidation**: 792 ROIs
- **ggo**: 404 ROIs
- **honeycombing**: 407 ROIs
- **reticulation**: 525 ROIs

### Disease Pattern Distribution
- **AHP**: 10 files
- **NSIP**: 14 files
- **OP**: 49 files
- **UIP**: 23 files

### Overall Axial Distribution
- **central**: 2 files
- **diffuse_scattered**: 26 files
- **peripheral**: 68 files

### Overall Manifestation
- **ggo_consolidation_mixed**: 59 files
- **subpleural**: 24 files
- **subpleural_omitted**: 13 files


## Term Translations

### Axial Distribution Terms
- **外周** → peripheral
- **中心** → central
- **弥漫/散在** → diffuse_scattered
- **弥漫分布** → diffuse
- **外周分布-胸膜下** → peripheral_subpleural
- **外周分布-其他** → peripheral_other
- **外周胸膜下省略+中心** → peripheral_subpleural_central
- **中心分布** → central

### Manifestation Terms
- **胸膜下** → subpleural
- **胸膜下省略** → subpleural_omitted
- **GGO实变不区分** → ggo_consolidation_mixed
- **其他** → other

## Usage Examples

### Python
```python
import json

# Read JSONL file
data = []
with open('processed_labels.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        data.append(json.loads(line.strip()))

# Access data
for item in data:
    print(f"File: {item['filename']}")
    print(f"Disease: {item['disease_pattern']}")
    print(f"Axial Distribution: {item['overall_axial_distribution']}")
    print(f"Manifestation: {item['overall_manifestation']}")
    print(f"ROI Count: {len(item['roi_layers'])}")
    print("---")
```

### Pandas
```python
import pandas as pd
import json

# Load as DataFrame
data = []
with open('processed_labels.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        data.append(json.loads(line.strip()))

df = pd.DataFrame(data)
print(df.head())
```

## Processing Script

The processing script is `process_labels.py`. To regenerate results:

```bash
python3 process_labels.py
```

## Notes

1. All missing values and null data have been appropriately handled
2. Overall file features are based on statistical analysis of all ROIs in that file
3. When multiple axial distributions or manifestations exist in a file, the most frequent one is selected as the overall feature
4. Data encoding uses UTF-8 to ensure proper Chinese character display
5. ROI information is preserved as parallel lists for efficient processing
6. All Chinese terms have been translated to English for international compatibility
