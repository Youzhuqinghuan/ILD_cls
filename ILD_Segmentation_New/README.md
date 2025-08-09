# ILD分割模块

本模块实现了完整的ILD（间质性肺疾病）分割解决方案，包括数据预处理、数据集类和训练支持。已成功处理97个患者的2061个切片数据，并实现了支持肺分割驱动边界框生成和背景切片处理的数据集类。

## 数据处理完成状态

✅ **数据预处理已完成** - 2025年8月6日
- 总计处理：2869个切片（包含808个背景切片）
- 训练集：2351个切片（1646病变 + 705背景）
- 验证集：258个切片（207病变 + 51背景）  
- 测试集：260个切片（208病变 + 52背景）
- 病变类型分布：honeycombing(430), reticulation(547), consolidation(811), GGO(273)
- 背景切片分布：训练集30.0%，验证集19.8%，测试集20.0%

## 主要功能

### 数据预处理模块 (data_preprocessing.py)
1. **CT、标签、肺分割文件的统一读取和预处理**
2. **基于病变类型的切片级数据划分**
3. **智能背景切片识别和处理**：自动识别无病变切片并按解剖位置分层采样
4. **生成标准化的npy格式训练数据**
5. **肺分割驱动的智能边界框生成**
6. **完整的元数据记录和统计报告**

### 数据集类模块 (dataset.py)
1. **支持肺分割驱动的边界框生成**：完全摆脱对Ground Truth的依赖
2. **背景切片训练支持**：处理无病变切片，提升模型鲁棒性
3. **平衡采样机制**：控制背景和病变切片的训练比例
4. **SAM2兼容格式**：直接输出1024×1024分辨率的训练数据
5. **智能标签处理**：随机选择病变类型，增强训练多样性

## 核心特性

- ✅ **切片级数据划分**：基于病变类型进行平衡的数据集划分
- ✅ **背景切片处理**：智能识别和处理无病变切片，提升模型鲁棒性
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
├── data_preprocessing.py    # 数据预处理模块
├── dataset.py              # 数据集类模块
├── example_usage.py        # 使用示例和测试
├── config.yaml             # 配置文件
├── DATASET_README.md       # 数据集详细文档
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
├── train/                  # 训练集 (2351个切片：1646病变 + 705背景)
│   ├── imgs/              # 图像文件
│   │   ├── CT_UIP 14_slice_066.npy      # 病变切片
│   │   ├── CT_AHP 10_slice_134.npy      # 病变切片
│   │   ├── CT_AHP 10_slice_042_bg.npy   # 背景切片
│   │   └── ...
│   ├── gts/               # 标签文件
│   │   ├── CT_UIP 14_slice_066.npy      # 病变标签
│   │   ├── CT_AHP 10_slice_134.npy      # 病变标签
│   │   ├── CT_AHP 10_slice_042_bg.npy   # 背景标签（全零）
│   │   └── ...
│   └── lungs/             # 肺分割文件
│       ├── CT_UIP 14_slice_066.npy      # 肺分割
│       ├── CT_AHP 10_slice_134.npy      # 肺分割
│       ├── CT_AHP 10_slice_042_bg.npy   # 背景切片肺分割
│       └── ...
├── val/                   # 验证集 (258个切片：207病变 + 51背景)
│   ├── imgs/
│   ├── gts/
│   └── lungs/
├── test/                  # 测试集 (260个切片：208病变 + 52背景)
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
- 总切片数和各数据集分布（包含背景切片统计）
- 病变类型分布统计
- 背景切片识别和分布统计
- 按患者的切片分布
- 处理成功率等质量指标

## 数据质量保证

- ✅ **完整性验证**：所有2869个切片（2061病变 + 808背景）均成功处理
- ✅ **格式标准化**：统一的256×256分辨率和npy格式
- ✅ **元数据完整**：每个切片都有详细的处理记录，包含背景切片标识
- ✅ **平衡划分**：各病变类型在训练/验证/测试集中均衡分布
- ✅ **背景切片质量**：智能识别算法确保背景切片的有效性和代表性

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

## 数据集类使用说明

### ILDDataset类特性

`ILDDataset`类是专为ILD分割任务设计的PyTorch数据集类，具有以下核心特性：

#### 1. 肺分割驱动的边界框生成
```python
from dataset import ILDDataset, create_dataloader

# 创建数据集实例
dataset = ILDDataset(
    data_root="data/preprocessed",
    split="train",
    bbox_shift=20,  # 边界框随机扰动范围
    background_sample_ratio=0.3,  # 背景切片采样比例
    enable_background_training=True  # 启用背景切片训练
)
```

#### 2. 背景切片处理
- **自动识别**：基于元数据自动识别背景切片（无病变）
- **平衡采样**：通过`WeightedRandomSampler`控制背景和病变切片比例
- **智能标签**：背景切片返回全零标签，病变切片随机选择病变类型

#### 3. 数据加载器创建
```python
# 创建训练数据加载器
train_loader = create_dataloader(
    data_root="data/preprocessed",
    split="train",
    batch_size=8,
    num_workers=4,
    background_sample_ratio=0.3
)

# 创建验证数据加载器（不启用背景训练）
val_loader = create_dataloader(
    data_root="data/preprocessed",
    split="val",
    batch_size=8,
    enable_background_training=False
)
```

#### 4. 数据格式
每个样本返回以下数据：
- `img_1024`: 图像张量 [3, 1024, 1024]，SAM2输入格式
- `gt2d`: 标签张量 [1, 256, 256]，二值分割标签
- `bbox`: 边界框张量 [4]，基于肺分割生成的ROI
- `filename`: 文件名字符串
- `is_background`: 背景标志张量 [1]，指示是否为背景切片

#### 5. 数据集统计
```python
# 获取数据集统计信息
stats = dataset.get_statistics()
print(f"总切片数: {stats['total_files']}")
print(f"背景切片比例: {stats['background_ratio']:.2%}")
print(f"数据集分布: {stats['dataset_stats']}")
```

### 关键改进对比

| 特性 | 旧数据集 (NpyDataset) | 新数据集 (ILDDataset) |
|------|---------------------|---------------------|
| 边界框生成 | 基于Ground Truth | 基于肺分割 |
| 背景切片 | 完全忽略 | 支持训练和推理 |
| 数据文件 | 仅图像和标签 | 图像、标签、肺分割 |
| 标签处理 | 固定病变类型 | 随机选择病变类型 |
| 元数据支持 | 无 | 完整元数据管理 |
| 平衡采样 | 无 | 支持背景/病变平衡 |

## 训练模块说明

### 训练程序特性 (train.py)

基于SAM2架构的ILD分割模型训练程序，具有以下核心特性：

#### 1. 模型架构
```python
class MedSAM2(nn.Module):
    """MedSAM2模型类，封装SAM2用于医学图像分割"""
    def __init__(self, model):
        super().__init__()
        self.sam2_model = model
        # 冻结prompt encoder
        for param in self.sam2_model.sam_prompt_encoder.parameters():
            param.requires_grad = False
```

#### 2. 加权损失函数
- **平衡背景和病变类别**：通过加权BCE和Dice损失处理类别不平衡
- **可配置权重**：支持背景权重、病变权重、Dice权重、CE权重的独立配置
- **自动正负样本平衡**：使用pos_weight机制平衡正负样本

```python
class WeightedLoss(nn.Module):
    def __init__(self, background_weight=0.5, lesion_weight=1.0, 
                 dice_weight=1.0, ce_weight=1.0):
        # 创建Dice损失和BCE损失
        # 自动计算pos_weight平衡正负样本
```

#### 3. 学习率调度策略
- **线性Warmup + 余弦退火**：使用SequentialLR组合调度器
- **标准化实现**：基于PyTorch官方调度器，支持状态保存和恢复
- **可配置参数**：warmup比例、最小学习率比例等可通过配置文件调整

```python
# 创建组合调度器
warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
    optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps
)
cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=total_steps - warmup_steps, eta_min=min_lr
)
scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps]
)
```

#### 4. 训练监控和指标
- **分类别指标计算**：分别计算背景和病变切片的性能指标
- **SwanLab集成**：实时监控训练过程，记录损失、学习率、指标等
- **详细日志记录**：完整的训练日志，包含每轮的详细指标

```python
# 分别计算背景和病变切片的指标
background_mask = is_background.squeeze() > 0.5
lesion_mask = ~background_mask

if torch.any(background_mask):
    bg_metrics = calculate_metrics(medsam_pred[background_mask], gt2D[background_mask])
if torch.any(lesion_mask):
    lesion_metrics = calculate_metrics(medsam_pred[lesion_mask], gt2D[lesion_mask])
```

#### 5. 检查点管理
- **完整状态保存**：模型、优化器、调度器状态的完整保存
- **最佳模型跟踪**：基于验证损失自动保存最佳模型
- **训练恢复**：支持从检查点恢复训练，保持状态一致性

### 训练配置说明

关键训练参数（`config.yaml`）：

```yaml
training:
  # 模型配置
  model:
    name: "sam2"
    config_file: "sam2_hiera_t.yaml"
    checkpoint: "./checkpoints/sam2_hiera_tiny.pt"
  
  # 训练参数
  params:
    epochs: 5
    batch_size: 16
    learning_rate: 1e-5
    weight_decay: 1e-4
  
  # 学习率调度器
  scheduler:
    type: "cosine_annealing"
    warmup_ratio: 0.1
    min_lr_ratio: 0.01
    warmup_type: "linear"
  
  # 损失函数
  loss:
    dice_weight: 1.0
    ce_weight: 1.0
    background_weight: 0.5
    lesion_weight: 1.0
```

### 训练使用示例

```bash
# 基本训练
python train.py -c config.yaml

# 从检查点恢复训练
python train.py -c config.yaml --resume ./workspace/bs16_lr1e-5_ep5_20250807_162718/latest_checkpoint.pth

# 使用自定义配置
python train.py -c custom_config.yaml
```

### 训练输出结构

```
workspace/
└── bs16_lr1e-5_ep5_20250807_162718/  # 实验目录
    ├── logs/
    │   └── train_20250807_162718.log    # 训练日志
    ├── config.yaml                      # 配置文件副本
    ├── train.py                         # 训练脚本副本
    ├── data_sanitycheck.png            # 数据检查可视化
    ├── best_model.pth                  # 最佳模型
    ├── latest_checkpoint.pth           # 最新检查点
    └── loss_curves.png                 # 损失曲线图
```

### 训练特点

#### 技术优势
- **标准化调度器**：使用PyTorch官方调度器，确保稳定性
- **状态一致性**：完整的检查点机制，支持无缝恢复训练
- **代码简洁性**：相比手动实现减少约15行复杂条件代码
- **可维护性**：模块化设计，易于扩展和修改

#### 训练策略
- **仅训练关键组件**：只训练image encoder和mask decoder，冻结prompt encoder
- **背景切片支持**：完整支持背景切片训练，提升模型鲁棒性
- **平衡采样**：通过数据集类的平衡采样机制控制训练比例
- **多指标监控**：precision、recall、specificity、F1、Dice等全面指标

## 后续应用

本模块可直接用于：
- **MedSAM2模型训练**：标准化的图像、标签和肺分割格式
- **深度学习框架**：PyTorch/TensorFlow数据加载
- **医学图像分析**：ILD病变分割和分类任务
- **ROI引导分割**：使用肺分割信息优化分割性能
- **模型评估**：独立的测试集用于性能评估
- **背景检测**：支持"无病变"状态的识别和处理