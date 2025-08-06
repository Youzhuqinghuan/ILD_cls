# ILD分割数据预处理模块

本模块实现了完整的ILD（间质性肺疾病）分割数据预处理流程，已成功处理97个患者的2061个切片数据。

## 数据处理完成状态

✅ **数据预处理已完成** - 2025年8月6日
- 总计处理：2061个切片
- 训练集：1646个切片
- 验证集：207个切片  
- 测试集：208个切片
- 病变类型分布：honeycombing(430), reticulation(547), consolidation(811), GGO(273)

## 主要功能

1. **CT、标签、肺分割文件的统一读取和预处理**
2. **基于病变类型的切片级数据划分**
3. **生成标准化的npy格式训练数据**
4. **肺分割驱动的智能边界框生成**
5. **完整的元数据记录和统计报告**

## 核心特性

- ✅ **切片级数据划分**：基于病变类型进行平衡的数据集划分
- ✅ **肺分割集成**：基于肺分割的边界框生成，优化ROI提取
- ✅ **标准化处理**：统一的256×256分辨率和数据格式
- ✅ **完整元数据**：记录每个切片的详细信息，包括病变类型、边界框等

## 数据处理流程

### 1. 数据加载与验证
- 验证CT图像目录：`../dataset/images`
- 验证标签目录：`../dataset/labels`  
- 验证肺分割目录：`../dataset/lungs`
- 加载标注文件：`../dataset/annotations/processed/processed_labels.jsonl`
- 加载病变分析：`../dataset/patient_lesion_analysis.json`

### 2. 数据划分策略
- **基于病变类型的切片级划分**：确保各病变类型在训练/验证/测试集中的平衡分布
- **划分比例**：训练集80%，验证集10%，测试集10%
- **病变类型分布**：
  - honeycombing: 训练344，验证43，测试43
  - reticulation: 训练437，验证55，测试55  
  - consolidation: 训练648，验证81，测试82
  - GGO: 训练217，验证28，测试28

### 3. 数据预处理步骤
1. **CT图像处理**：
   - 加载原始DICOM/NIfTI格式CT数据
   - 应用窗位窗宽设置优化显示
   - HU值归一化到0-255范围
   
2. **肺分割处理**：
   - 加载对应的肺分割mask
   - 保持完整的肺分割信息作为ROI参考
   - 为MedSAM2提供边界框信息
   
3. **标准化处理**：
   - 对完整切片进行标准化处理（不裁剪）
   - 调整尺寸到256×256标准分辨率
   - 生成3通道图像格式
   
4. **标签和肺分割处理**：
   - 加载对应的分割标签
   - 应用相同的缩放变换到标签和肺分割
   - 保持标签值和肺分割的完整性

### 4. 输出数据格式
- **图像文件**：256×256×3，uint8格式，值范围0-255
- **标签文件**：256×256，uint16格式，0为背景，其他值为病变类型
- **肺分割文件**：256×256，uint8格式，0为背景，1为肺区域
- **文件命名**：`患者ID_切片编号.npy`

## 文件结构

```
ILD_Segmentation_New/
├── data_preprocessing.py    # 主要的数据预处理模块
├── config.yaml             # 配置文件
├── requirements.txt        # 依赖包列表
├── data/
│   └── preprocessed/       # 预处理输出目录
│       ├── train/          # 训练集数据
│       ├── val/            # 验证集数据
│       ├── test/           # 测试集数据
│       └── *.json          # 元数据文件
└── README.md              # 本文件
```

## 最终数据结构

预处理完成后生成的标准化数据集结构：

```
data/preprocessed/
├── train/                  # 训练集 (1646个切片)
│   ├── imgs/              # 图像文件
│   │   ├── CT_UIP 14_slice_066.npy
│   │   ├── CT_AHP 10_slice_134.npy
│   │   └── ...
│   ├── gts/               # 标签文件
│   │   ├── CT_UIP 14_slice_066.npy
│   │   ├── CT_AHP 10_slice_134.npy
│   │   └── ...
│   └── lungs/             # 肺分割文件
│       ├── CT_UIP 14_slice_066.npy
│       ├── CT_AHP 10_slice_134.npy
│       └── ...
├── val/                   # 验证集 (207个切片)
│   ├── imgs/
│   ├── gts/
│   └── lungs/
├── test/                  # 测试集 (208个切片)
│   ├── imgs/
│   ├── gts/
│   └── lungs/
├── slice_splits.json      # 切片划分信息
├── slice_metadata.json    # 详细切片元数据
├── slice_metadata_full.json  # 完整元数据
└── preprocessing_statistics.json  # 统计报告
```

## 数据使用说明

### 加载数据示例

```python
import numpy as np
import json

# 加载图像、标签和肺分割
img = np.load('data/preprocessed/train/imgs/CT_UIP 14_slice_066.npy')
gt = np.load('data/preprocessed/train/gts/CT_UIP 14_slice_066.npy')
lung = np.load('data/preprocessed/train/lungs/CT_UIP 14_slice_066.npy')

print(f"图像形状: {img.shape}")  # (256, 256, 3)
print(f"标签形状: {gt.shape}")   # (256, 256)
print(f"肺分割形状: {lung.shape}")  # (256, 256)
print(f"图像范围: [{img.min()}, {img.max()}]")  # [0, 255]
print(f"标签值: {np.unique(gt)}")  # [0, 1, 2, ...]
print(f"肺分割值: {np.unique(lung)}")  # [0, 1]

# 加载元数据
with open('data/preprocessed/slice_metadata.json', 'r') as f:
    metadata = json.load(f)
    
# 加载统计信息
with open('data/preprocessed/preprocessing_statistics.json', 'r') as f:
    stats = json.load(f)
    print(f"总切片数: {stats['total_slices']}")
```

### 配置文件说明

关键配置参数（`config.yaml`）：

```yaml
data_paths:
  ct_dir: "../dataset/images"          # CT图像目录
  label_dir: "../dataset/labels"       # 标签目录  
  lung_dir: "../dataset/lungs"         # 肺分割目录
  annotations_file: "../dataset/annotations/processed/processed_labels.jsonl"
  output_dir: "data/preprocessed"       # 输出目录

processing:
  target_size: [256, 256]             # 目标分辨率
  window_level: 40                     # CT窗位
  window_width: 400                    # CT窗宽
  bbox_padding: 10                     # 边界框padding
```

## 元数据文件说明

### slice_metadata.json
记录每个切片的详细处理信息：
```json
[
  {
    "status": "success",
    "original_size": [512, 512],
    "final_size": [256, 256],
    "lung_available": true,
    "lesion_types": [4],
    "primary_lesion": 4,
    "has_lesion": true,
    "split": "train",
    "patient_id": "CT_UIP 14",
    "slice_name": "slice_066"
  }
]
```

### preprocessing_statistics.json
包含完整的数据统计信息：
- 总切片数和各数据集分布
- 病变类型分布统计
- 按患者的切片分布
- 处理成功率等质量指标

## 数据质量保证

- ✅ **完整性验证**：所有2061个切片均成功处理
- ✅ **格式标准化**：统一的256×256分辨率和npy格式
- ✅ **元数据完整**：每个切片都有详细的处理记录
- ✅ **平衡划分**：各病变类型在训练/验证/测试集中均衡分布

## 技术特点

### 处理优化
- **按患者分组处理**：减少重复I/O操作
- **完整切片处理**：保持原始空间关系，不进行裁剪
- **肺分割保留**：为MedSAM2提供ROI参考信息
- **多进程并行**：提高处理效率
- **内存优化**：渐进式处理避免内存溢出

### 数据标准化
- **窗位窗宽处理**：优化CT图像显示
- **HU值归一化**：统一数据范围
- **尺寸标准化**：256×256分辨率
- **格式统一**：npy格式便于快速加载
- **三模态输出**：图像、标签、肺分割同步处理

## 使用建议

1. **直接使用**：数据已预处理完成，可直接用于模型训练
2. **数据加载**：使用numpy直接加载npy文件
3. **元数据查询**：通过JSON文件获取切片详细信息
4. **质量检查**：查看统计报告了解数据分布

## 后续应用

本预处理数据可直接用于：
- **MedSAM2模型训练**：标准化的图像、标签和肺分割格式
- **深度学习框架**：PyTorch/TensorFlow数据加载
- **医学图像分析**：ILD病变分割和分类任务
- **ROI引导分割**：使用肺分割信息优化分割性能
- **模型评估**：独立的测试集用于性能评估