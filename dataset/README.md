# ILD 间质性肺疾病数据集

## 概述

本目录包含 ILD（间质性肺疾病）分类项目的标注数据和处理脚本。数据集用于训练和评估基于 HRCT 影像的 ILD 模式分类模型。

## 文件结构

```
dataset/
├── README.md                          # 本文档
├── images/                            # CT 图像文件（.nii.gz 格式）
│   ├── CT_UIP1_0000.nii.gz           # UIP 病例 CT 图像
│   ├── CT_NSIP1_0000.nii.gz          # NSIP 病例 CT 图像
│   └── ...                           # 其他 CT 图像文件
├── labels/                            # 病变分割标签（.nii.gz 格式）
│   ├── CT_UIP1.nii.gz                # UIP 病例分割标签
│   ├── CT_NSIP1.nii.gz               # NSIP 病例分割标签
│   └── ...                           # 其他分割标签文件
└── annotations/
    ├── labels.csv                     # 原始标注数据
    ├── process_labels.py              # 数据处理脚本
    └── processed/
        ├── README.md                  # 详细的英文说明文档
        └── processed_labels.jsonl     # 处理后的标准化数据
```

## 数据说明

### CT 图像数据 (images/)

包含所有病例的 3D HRCT 图像，以 `.nii.gz` 格式存储。文件命名格式为 `CT_{病例名}_0000.nii.gz`。

数据分为两类：
1. **有分割标签的 CT 数据**：对应 `labels.csv` 中记录的病例，同时在 `labels/` 目录下有对应的分割标签
2. **无分割标签的 CT 数据**：仅有 CT 图像，无对应的分割标签

### 病变分割标签 (labels/)

包含部分病例的病变分割掩膜，以 `.nii.gz` 格式存储。文件命名格式为 `CT_{病例名}.nii.gz`。
只有在 `labels.csv` 中有记录的病例才有对应的分割标签。

### 原始标注数据 (labels.csv)

包含 HRCT 影像的详细标注信息，主要字段包括：
- **文件名**: 病例标识符（如 "UIP 1", "NSIP 5"）
- **HRCT日期**: 影像采集日期
- **标记所在层面**: ROI 所在的切片序号
- **异常征象**: 病变类型（网格、蜂窝、GGO、实变等）
- **轴向分布**: 病变在轴向的分布模式（外周、中心、弥漫等）
- **整体征象**: 胸膜下特征等

### 处理后数据 (processed_labels.jsonl)

经过清洗和标准化的 JSONL 格式数据，每行包含一个病例的完整信息：

```json
{
  "filename": "UIP 1",
  "disease_pattern": "UIP", 
  "overall_axial_distribution": "peripheral",
  "overall_manifestation": "subpleural",
  "abnormal_manifestation_presence": [1, 1, 0, 0],
  "roi_layers": [150, 136, 126, ...],
  "roi_axial_distributions": ["peripheral", "peripheral", ...],
  "roi_manifestations": ["subpleural", "subpleural", ...]
}
```

## 数据统计

### 总体数据分布

#### 有分割标签的 CT 数据
- **UIP**: 1-23 (23 个病例)
- **NSIP**: 1-19 (14 个病例)
- **OP**: 1-50 (49 个病例)
- **AHP**: 1-11 (10 个病例)
- **总计**: 96 个病例，均有对应的分割标签和详细标注

#### 无分割标签的 CT 数据
- **UIP**: 24-88 (65 个病例)
- **NSIP**: 20-40 (21 个病例)
- **CHP**: 1-15 (15 个病例)
- **UNC**: 1-45 (45 个病例)
- **Normal**: 1-80 (80 个病例)
- **总计**: 226 个病例，仅有 CT 图像

#### 标注数据统计 (基于 labels.csv)
- **总文件数**: 96 个
- **总 ROI 数**: 2,121 个
- **平均每文件 ROI 数**: 22.09 个

### 疾病模式分布

根据临床分类标准，疾病模式分为 5 大类：

- **UIP** (寻常型间质性肺炎): 23 个文件
- **NSIP** (非特异性间质性肺炎): 14 个文件  
- **OP** (机化性肺炎): 49 个文件
- **Other** (其他类型): 70 个文件
  - AHP (急性过敏性肺炎)
  - CHP (慢性过敏性肺炎) 
  - UNC (无法分类)
- **Normal** (正常): 无标注数据，仅在无分割标签数据中存在

### 异常征象统计（ROI 级别）
- **实变 (consolidation)**: 792 个 ROI
- **磨玻璃影 (GGO)**: 404 个 ROI
- **蜂窝影 (honeycombing)**: 407 个 ROI
- **网格影 (reticulation)**: 525 个 ROI

### 轴向分布统计
- **外周 (peripheral)**: 68 个文件
- **弥漫/散在 (diffuse_scattered)**: 26 个文件
- **中心 (central)**: 2 个文件

## 使用方法

### 数据处理

运行处理脚本来重新生成标准化数据：

```bash
cd annotations
python process_labels.py
```

### 数据读取示例

```python
import json

# 读取处理后的数据
data = []
with open('annotations/processed/processed_labels.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        data.append(json.loads(line.strip()))

# 访问数据
for item in data:
    print(f"病例: {item['filename']}")
    print(f"疾病模式: {item['disease_pattern']}")
    print(f"轴向分布: {item['overall_axial_distribution']}")
    print(f"整体征象: {item['overall_manifestation']}")
    print(f"ROI 数量: {len(item['roi_layers'])}")
    print("---")
```

## 术语对照表

### 轴向分布术语
- **外周** → peripheral
- **中心** → central
- **弥漫/散在** → diffuse_scattered

### 整体征象术语
- **胸膜下** → subpleural
- **胸膜下省略** → subpleural_omitted  
- **GGO实变不区分** → ggo_consolidation_mixed (表示该病例分类无需考虑整体征象因素)

### 异常征象术语
- **蜂窝** → honeycombing
- **网格** → reticulation
- **实变** → consolidation
- **磨玻璃影** → GGO (Ground-Glass Opacity)

## 注意事项

1. 原始数据使用 UTF-8 编码，包含中文标注
2. 处理后的数据已转换为英文术语，便于模型训练
3. 每个文件的整体特征基于该文件内所有 ROI 的统计分析得出
4. 异常征象存在标记用 4 元素布尔列表表示：[蜂窝, 网格, 实变, GGO]
5. 详细的处理逻辑和英文说明请参考 `processed/README.md`

## 更新日志

- 初始版本：包含 96 个病例的完整标注数据
- 数据清洗：标准化术语，转换为英文，生成 JSONL 格式