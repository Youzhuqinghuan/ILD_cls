#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Medical Image Annotation Data Processing Program
Extract and simplify information from CSV file, output as JSONL format
"""

import pandas as pd
import json
import re
from collections import Counter
import os

def extract_disease_pattern(filename):
    """
    Extract disease pattern from filename
    
    Args:
        filename: File identifier (e.g., 'UIP 5', 'NSIP 12', 'AHP 1')
    
    Returns:
        str: Disease pattern (UIP, NSIP, AHP, OP, etc.)
    """
    # Remove numbers and spaces to get disease pattern
    pattern = re.sub(r'[\d\s]+', '', filename).strip()
    return pattern if pattern else "Unknown"

def translate_axial_distribution(chinese_term):
    """
    Translate Chinese axial distribution terms to English
    
    Args:
        chinese_term: Chinese term
    
    Returns:
        str: English term
    """
    translation_map = {
        '外周': 'peripheral',
        '中心': 'central', 
        '弥漫/散在': 'diffuse_scattered',
        '弥漫分布': 'diffuse',
        '外周分布-胸膜下': 'peripheral_subpleural',
        '外周分布-其他': 'peripheral_other',
        '外周胸膜下省略+中心': 'peripheral_subpleural_central',
        '中心分布': 'central'
    }
    return translation_map.get(chinese_term, chinese_term)

def translate_manifestation(chinese_term):
    """
    Translate Chinese manifestation terms to English
    
    Args:
        chinese_term: Chinese term
    
    Returns:
        str: English term
    """
    translation_map = {
        '胸膜下': 'subpleural',
        '胸膜下省略': 'subpleural_omitted',
        'GGO实变不区分': 'ggo_consolidation_mixed',
        '其他': 'other'
    }
    return translation_map.get(chinese_term, chinese_term)

def process_labels_csv(csv_file_path):
    """
    Process annotation CSV file and extract simplified information
    
    Args:
        csv_file_path: Path to CSV file
    
    Returns:
        tuple: (list of processed data dictionaries, global abnormal manifestation counts)
    """
    # Read CSV file
    df = pd.read_csv(csv_file_path, encoding='utf-8')
    
    # Required columns
    required_columns = [
        '图像清洗后编码',  # filename
        '标记所在层面',    # ROI layer
        '左右肺整体轴向分布',  # axial distribution
        '左右肺整体征象'     # manifestation
    ]
    
    # Check if columns exist
    for col in required_columns:
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found in CSV file")
            print(f"Available columns: {list(df.columns)}")
            return None, None
    
    # Filter out rows with empty filename
    df_filtered = df.dropna(subset=['图像清洗后编码'])
    
    # Initialize global abnormal manifestation counts (ROI level)
    global_abnormal_counts = {
        'honeycombing': 0,
        'reticulation': 0,
        'consolidation': 0,
        'ggo': 0
    }
    
    # Process by filename groups
    result = []
    
    for filename, group in df_filtered.groupby('图像清洗后编码'):
        # Extract disease pattern
        disease_pattern = extract_disease_pattern(filename)
        
        # Count axial distributions
        axial_distribution = group['左右肺整体轴向分布'].dropna()
        axial_counter = Counter(axial_distribution)
        
        # Count manifestations
        manifestation = group['左右肺整体征象'].dropna()
        manifestation_counter = Counter(manifestation)
        
        # Get most frequent axial distribution and manifestation
        most_common_axial = axial_counter.most_common(1)[0][0] if axial_counter else "unknown"
        most_common_manifestation = manifestation_counter.most_common(1)[0][0] if manifestation_counter else "unknown"
        
        # Translate to English
        most_common_axial_en = translate_axial_distribution(most_common_axial)
        most_common_manifestation_en = translate_manifestation(most_common_manifestation)
        
        # Check for specific abnormal manifestations in the file (for file-level boolean)
        abnormal_manifestations_in_file = set()
        
        # Count abnormal manifestations at ROI level and update global counts
        if '异常征象' in group.columns:
            for manifestation in group['异常征象'].dropna():
                if manifestation == '蜂窝':
                    abnormal_manifestations_in_file.add('honeycombing')
                    global_abnormal_counts['honeycombing'] += 1
                elif manifestation == '网格':
                    abnormal_manifestations_in_file.add('reticulation')
                    global_abnormal_counts['reticulation'] += 1
                elif manifestation == '实变':
                    abnormal_manifestations_in_file.add('consolidation')
                    global_abnormal_counts['consolidation'] += 1
                elif manifestation == 'GGO':
                    abnormal_manifestations_in_file.add('ggo')
                    global_abnormal_counts['ggo'] += 1
        
        # Create boolean list for abnormal manifestations (蜂窝, 网格, 实变, GGO)
        abnormal_manifestation_presence = [
            1 if 'honeycombing' in abnormal_manifestations_in_file else 0,
            1 if 'reticulation' in abnormal_manifestations_in_file else 0,
            1 if 'consolidation' in abnormal_manifestations_in_file else 0,
            1 if 'ggo' in abnormal_manifestations_in_file else 0
        ]
        
        # Collect ROI layer information
        roi_data = []
        
        for _, row in group.iterrows():
            if pd.notna(row['标记所在层面']):
                # Translate axial distribution
                axial_dist = row['左右肺整体轴向分布'] if pd.notna(row['左右肺整体轴向分布']) else "unknown"
                
                # Translate manifestation
                manifest = row['左右肺整体征象'] if pd.notna(row['左右肺整体征象']) else "unknown"
                
                roi_data.append({
                    'layer': int(row['标记所在层面']) - 1,  # 减1
                    'axial_dist': translate_axial_distribution(axial_dist),
                    'manifestation': translate_manifestation(manifest)
                })
        
        # 按layer从小到大排序
        roi_data.sort(key=lambda x: x['layer'])
        
        # 提取排序后的数据
        roi_layers = [item['layer'] for item in roi_data]
        roi_axial_distributions = [item['axial_dist'] for item in roi_data]
        roi_manifestations = [item['manifestation'] for item in roi_data]
        
        # Build file information
        file_info = {
            "filename": filename,
            "disease_pattern": disease_pattern,
            "overall_axial_distribution": most_common_axial_en,
            "overall_manifestation": most_common_manifestation_en,
            "abnormal_manifestation_presence": abnormal_manifestation_presence,
            "roi_layers": roi_layers,
            "roi_axial_distributions": roi_axial_distributions,
            "roi_manifestations": roi_manifestations
        }
        
        result.append(file_info)
    
    return result, global_abnormal_counts

def save_jsonl(data, output_file):
    """
    Save data as JSONL format
    
    Args:
        data: Processed data list
        output_file: Output file path
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')
    print(f"JSONL file saved to: {output_file}")

def generate_statistics(data, global_abnormal_counts):
    """
    Generate statistics from processed data
    
    Args:
        data: Processed data list
        global_abnormal_counts: Global abnormal manifestation counts at ROI level
    
    Returns:
        dict: Statistics dictionary
    """
    stats = {
        'total_files': len(data),
        'disease_patterns': {},
        'overall_axial_distributions': {},
        'overall_manifestations': {},
        'abnormal_manifestation_stats': global_abnormal_counts.copy(),
        'roi_statistics': {
            'total_rois': 0,
            'avg_rois_per_file': 0,
            'min_rois_per_file': float('inf'),
            'max_rois_per_file': 0
        }
    }
    
    roi_counts = []
    
    for item in data:
        # Disease pattern statistics
        pattern = item['disease_pattern']
        stats['disease_patterns'][pattern] = stats['disease_patterns'].get(pattern, 0) + 1
        
        # Overall axial distribution statistics
        axial = item['overall_axial_distribution']
        stats['overall_axial_distributions'][axial] = stats['overall_axial_distributions'].get(axial, 0) + 1
        
        # Overall manifestation statistics
        manifest = item['overall_manifestation']
        stats['overall_manifestations'][manifest] = stats['overall_manifestations'].get(manifest, 0) + 1
        
        # ROI statistics
        roi_count = len(item['roi_layers'])
        roi_counts.append(roi_count)
        stats['roi_statistics']['total_rois'] += roi_count
        stats['roi_statistics']['min_rois_per_file'] = min(stats['roi_statistics']['min_rois_per_file'], roi_count)
        stats['roi_statistics']['max_rois_per_file'] = max(stats['roi_statistics']['max_rois_per_file'], roi_count)
    
    if roi_counts:
        stats['roi_statistics']['avg_rois_per_file'] = sum(roi_counts) / len(roi_counts)
    
    return stats

def save_readme(stats, output_dir):
    """
    Generate and save README file with statistics
    
    Args:
        stats: Statistics dictionary
        output_dir: Output directory
    """
    readme_content = f"""# Medical Image Annotation Data Processing Results

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
{{
  "filename": "file_identifier",
  "disease_pattern": "disease_type",
  "overall_axial_distribution": "most_frequent_axial_distribution",
  "overall_manifestation": "most_frequent_manifestation",
  "abnormal_manifestation_presence": [int, int, int, int],
  "roi_layers": [layer_numbers],
  "roi_axial_distributions": ["axial_distribution_per_roi"],
  "roi_manifestations": ["manifestation_per_roi"]
}}
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
- **Total Files**: {stats['total_files']}
- **Total ROIs**: {stats['roi_statistics']['total_rois']}
- **Average ROIs per File**: {stats['roi_statistics']['avg_rois_per_file']:.2f}
- **Min ROIs per File**: {stats['roi_statistics']['min_rois_per_file']}
- **Max ROIs per File**: {stats['roi_statistics']['max_rois_per_file']}

### Abnormal Manifestation Statistics (ROI Level)
"""
    
    for manifestation, count in sorted(stats['abnormal_manifestation_stats'].items()):
        readme_content += f"- **{manifestation}**: {count} ROIs\n"
    
    readme_content += "\n### Disease Pattern Distribution\n"
    for pattern, count in sorted(stats['disease_patterns'].items()):
        readme_content += f"- **{pattern}**: {count} files\n"
    
    readme_content += "\n### Overall Axial Distribution\n"
    for dist, count in sorted(stats['overall_axial_distributions'].items()):
        readme_content += f"- **{dist}**: {count} files\n"
    
    readme_content += "\n### Overall Manifestation\n"
    for manifest, count in sorted(stats['overall_manifestations'].items()):
        readme_content += f"- **{manifest}**: {count} files\n"
    
    readme_content += f"""

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
    print(f"File: {{item['filename']}}")
    print(f"Disease: {{item['disease_pattern']}}")
    print(f"Axial Distribution: {{item['overall_axial_distribution']}}")
    print(f"Manifestation: {{item['overall_manifestation']}}")
    print(f"ROI Count: {{len(item['roi_layers'])}}")
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
"""
    
    readme_path = os.path.join(output_dir, 'README.md')
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    print(f"README file saved to: {readme_path}")

def main():
    """
    Main function
    """
    # Input file path
    csv_file_path = '/home/huchengpeng/workspace/dataset/annotations/labels.csv'
    
    # Output directory
    output_dir = '/home/huchengpeng/workspace/dataset/annotations/processed'
    
    print(f"Processing file: {csv_file_path}")
    
    # Process data
    processed_data, global_abnormal_counts = process_labels_csv(csv_file_path)
    
    if processed_data is None:
        print("Data processing failed")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save JSONL file
    jsonl_file = os.path.join(output_dir, 'processed_labels.jsonl')
    save_jsonl(processed_data, jsonl_file)
    
    # Generate statistics
    stats = generate_statistics(processed_data, global_abnormal_counts)
    
    # Save README with statistics
    save_readme(stats, output_dir)
    
    print("\n=== Processing Complete ===")
    print(f"Total files processed: {stats['total_files']}")
    print(f"Total ROIs: {stats['roi_statistics']['total_rois']}")
    print(f"Output directory: {output_dir}")
    
    # Show first few examples
    print("\n=== Sample Results ===")
    for i, item in enumerate(processed_data[:3]):
        print(f"\nFile {i+1}: {item['filename']}")
        print(f"  Disease Pattern: {item['disease_pattern']}")
        print(f"  Overall Axial Distribution: {item['overall_axial_distribution']}")
        print(f"  Overall Manifestation: {item['overall_manifestation']}")
        if 'abnormal_manifestation_presence' in item:
            print(f"  Abnormal Manifestation Presence: {item['abnormal_manifestation_presence']}")
        print(f"  ROI Count: {len(item['roi_layers'])}")
    
    # Print abnormal manifestation statistics
    if 'abnormal_manifestation_stats' in stats:
        print("\n=== Abnormal Manifestation Statistics ===")
        for manifestation, count in stats['abnormal_manifestation_stats'].items():
            print(f"  {manifestation}: {count} files")

if __name__ == "__main__":
    main()