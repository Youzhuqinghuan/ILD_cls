# 放射学描述符分析程序

基于规则的算法自动分析CT图像中病变的放射学描述符，包括轴向分布和整体征象。

## 功能特性

### 轴向分布分析
- **外周 (peripheral)**: 病变主要位于肺部外周区域 → peripheral
- **中心 (central)**: 病变主要位于肺部中心区域   → central
- **弥漫/散在 (diffuse_scattered)**: 病变在外周和中心区域均有分布 → diffuse_scattered

### 整体征象分析
- **胸膜下 (subpleural)**: 病变累及胸膜下区域（距胸膜≤3mm）  → subpleural
- **胸膜下省略 (subpleural_omitted)**: 病变未累及胸膜下区域 → subpleural_omitted

## 算法原理

### 预处理步骤
1. **整肺掩码生成**: 使用阈值分割和形态学操作从CT图像生成整肺掩码
2. **距离变换**: 计算肺内每个体素到胸膜表面的欧氏距离
3. **归一化**: 使用最大半径进行相对距离归一化

### 分类规则

#### 轴向分布判断
```python
# 参数设置
per_mm = 10        # 外周阈值（毫米）
per_rel = 0.20     # 外周相对阈值（20%）

# 外周体素判断
peripheral_voxels = (distance <= 10mm) OR (distance/max_radius <= 0.20)

# 分类规则
if peripheral_ratio >= 0.70:
    axial_distribution = "peripheral"
elif central_ratio >= 0.70:
    axial_distribution = "central"
else:
    axial_distribution = "diffuse_scattered"
```

#### 整体征象判断
```python
# 胸膜下阈值
subpleural_threshold = 3.0  # 毫米

# 分类规则
if min_distance_to_pleura <= 3mm:
    manifestation = "subpleural"
else:
    manifestation = "subpleural_omitted"
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 测试单个病例

```bash
# 测试NSIP1病例
python radiological_descriptor_analyzer.py --test_single NSIP1

# 测试UIP10病例
python radiological_descriptor_analyzer.py --test_single UIP10
```

### 2. 批量处理所有病例

```bash
# 使用默认路径
python radiological_descriptor_analyzer.py

# 自定义输入输出路径
python radiological_descriptor_analyzer.py \
    --images_dir /path/to/images \
    --labels_dir /path/to/labels \
    --output_dir /path/to/output
```

### 3. 参数说明

- `--images_dir`: CT图像目录（默认: `/home/huchengpeng/workspace/dataset/images`）
- `--labels_dir`: 病变掩码目录（默认: `/home/huchengpeng/workspace/dataset/labels`）
- `--output_dir`: 输出结果目录（默认: `/home/huchengpeng/workspace/Generate_Descriptors`）
- `--test_single`: 测试单个病例名称（如: `NSIP1`）

## 输入文件格式

程序期望以下文件命名格式：
- CT图像: `CT_{name}_0000.nii.gz`
- 病变掩码: `CT_{name}.nii.gz`

例如：
- `CT_NSIP1_0000.nii.gz` (CT图像)
- `CT_NSIP1.nii.gz` (对应的病变掩码)

## 输出格式

程序输出JSONL格式文件，每行包含一个病例的分析结果：

```json
{
  "case_name": "NSIP1",
  "axial_distribution": "peripheral",
  "manifestation": "subpleural",
  "axial_metrics": {
    "peripheral_ratio": 0.85,
    "central_ratio": 0.15,
    "mean_distance_mm": 8.5,
    "min_distance_mm": 0.5,
    "max_distance_mm": 25.3,
    "max_radius_mm": 45.2
  },
  "manifestation_metrics": {
    "min_distance_to_pleura_mm": 0.5,
    "subpleural_ratio": 0.35,
    "subpleural_involved": true
  }
}
```

## 输出文件

- 单个病例测试: `radiological_descriptors_{case_name}.jsonl`
- 批量处理: `radiological_descriptors_all.jsonl`

## 术语映射

根据 `/home/huchengpeng/workspace/dataset/annotations/processed/README.md` 中的定义：

### 轴向分布术语
- **外周** → `peripheral`
- **中心** → `central`
- **弥漫/散在** → `diffuse_scattered`

### 整体征象术语
- **胸膜下** → `subpleural`
- **胸膜下省略** → `subpleural_omitted`

## 算法验证

该算法基于多项医学研究验证：
- 单一"距胸膜距离"特征可有效区分外周炎性病灶和中心肺癌（AUC≈0.73）
- 可辅助UIP vs. NSIP判别
- 商业化纤维化定量软件也采用类似的距离图框架

## 性能

- 处理单例HRCT仅需数十秒（CPU）
- 可重现文献中的定量指标
- 支持批量处理多个病例

## 注意事项

1. 确保CT图像和病变掩码文件名匹配
2. 病变掩码应为二值图像（0和1）
3. CT图像应为标准的NIfTI格式
4. 程序会自动生成整肺掩码，无需额外提供

## 故障排除

### 常见错误

1. **Empty disease mask**: 病变掩码为空
   - 检查病变掩码文件是否正确
   - 确认掩码中包含非零值

2. **Failed to generate lung mask**: 无法生成肺部掩码
   - 检查CT图像质量
   - 调整阈值参数（默认-300HU）

3. **No disease voxels within lung mask**: 病变区域不在肺部范围内
   - 检查CT图像和病变掩码的配准
   - 确认两者空间坐标一致

### 调试建议

使用单个病例测试模式进行调试：
```bash
python radiological_descriptor_analyzer.py --test_single NSIP1
```

查看详细的处理信息和错误提示。

## 评估程序

### 功能介绍

`evaluate_descriptors.py` 程序用于评估算法预测结果的准确率，通过比较标注数据和算法预测结果来计算性能指标。

### 比较内容

1. **轴向分布**: 比较 `overall_axial_distribution`（标注）与 `axial_distribution`（预测）
2. **整体征象**: 比较 `overall_manifestation`（标注）与 `manifestation`（预测）

### 使用方法

#### 1. 基本评估

```bash
# 使用默认路径进行评估
python evaluate_descriptors.py
```

#### 2. 自定义文件路径

```bash
python evaluate_descriptors.py \
    --annotations /path/to/annotations.jsonl \
    --predictions /path/to/predictions.jsonl \
    --output /path/to/evaluation_results.json
```

#### 3. 显示详细结果

```bash
# 显示所有病例的详细比较结果
python evaluate_descriptors.py --detailed

# 只显示预测错误的病例
python evaluate_descriptors.py --detailed --errors-only
```

### 参数说明

- `--annotations`: 标注数据文件路径（默认: `/home/huchengpeng/workspace/dataset/annotations/processed/processed_labels.jsonl`）
- `--predictions`: 算法预测结果文件路径（默认: `/home/huchengpeng/workspace/Generate_Descriptors/radiological_descriptors_all.jsonl`）
- `--output`: 评估结果输出文件路径（默认: `/home/huchengpeng/workspace/Generate_Descriptors/evaluation_results.json`）
- `--detailed`: 显示详细的逐病例比较结果
- `--errors-only`: 只显示预测错误的病例（需与 `--detailed` 一起使用）

### 输入文件格式

#### 标注数据格式 (processed_labels.jsonl)
```json
{
  "filename": "NSIP 1",
  "disease_pattern": "NSIP",
  "overall_axial_distribution": "peripheral",
  "overall_manifestation": "subpleural_omitted"
}
```

#### 预测结果格式 (radiological_descriptors_all.jsonl)
```json
{
  "case_name": "NSIP1",
  "axial_distribution": "peripheral",
  "manifestation": "subpleural"
}
```

### 输出格式

#### 控制台输出示例
```
============================================================
放射学描述符评估结果
============================================================

1. 轴向分布 (Axial Distribution):
   正确预测: 35/37
   准确率: 94.59%

2. 整体征象 (Manifestation):
   正确预测: 28/37
   准确率: 75.68%

3. 总体准确率:
   正确预测: 63/74
   准确率: 85.14%
============================================================
```

#### JSON输出文件格式
```json
{
  "axial_distribution": {
    "correct": 35,
    "total": 37,
    "accuracy": 0.9459,
    "details": [
      {
        "case_name": "NSIP1",
        "annotation": "peripheral",
        "prediction": "peripheral",
        "correct": true
      }
    ]
  },
  "manifestation": {
    "correct": 28,
    "total": 37,
    "accuracy": 0.7568,
    "details": [
      {
        "case_name": "NSIP1",
        "annotation": "subpleural_omitted",
        "prediction": "subpleural",
        "correct": false
      }
    ]
  }
}
```

### 病例名称匹配

程序会自动处理标注数据和预测结果之间的命名差异：
- 标注数据: `"NSIP 1"`, `"UIP 10"`
- 预测结果: `"NSIP1"`, `"UIP10"`

### 评估指标

1. **准确率 (Accuracy)**: 正确预测数量 / 总预测数量
2. **详细比较**: 逐病例显示标注值、预测值和是否正确
3. **错误分析**: 可选择只显示预测错误的病例

### 使用示例

```bash
# 完整评估流程
# 1. 运行算法生成预测结果
python radiological_descriptor_analyzer.py

# 2. 评估预测准确率
python evaluate_descriptors.py --detailed

# 3. 查看详细的错误分析
python evaluate_descriptors.py --detailed --errors-only
```

### 注意事项

1. 确保标注数据和预测结果文件都存在
2. 程序会自动匹配病例名称，处理命名格式差异
3. 只有在两个文件中都存在的病例才会被评估
4. 评估结果会同时显示在控制台和保存到JSON文件