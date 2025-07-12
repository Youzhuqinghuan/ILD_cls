# 肺部分割程序

基于阈值和形态学方法的CT图像肺部分割程序，可以处理 `.nii.gz` 格式的CT图像。包含肺部分割和肺叶分析两个主要功能模块。

## 功能特点

### 肺部分割 (lung_segmentation.py)
- 使用阈值分割和形态学操作进行肺部分割
- 支持批量处理多个CT文件
- 自动保存肺部掩码和分割后的CT图像
- 可选的可视化功能
- 支持单文件测试模式

### 肺叶分析 (lung_lobe_analysis.py)
- 基于几何百分位数方法划分上下肺
- 计算病变在上下肺的分布和占比
- 支持体积计算（考虑像素间距）
- 输出详细的统计分析结果
- 支持JSONL格式结果保存

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 肺部分割 (lung_segmentation.py)

#### 1. 测试单个文件（推荐先测试）

```bash
# 测试 NSIP1 文件
python lung_segmentation.py --test_single CT_NSIP1_0000.nii.gz --visualize
```

#### 2. 批量处理所有文件

```bash
# 处理所有CT文件
python lung_segmentation.py

# 带可视化的处理
python lung_segmentation.py --visualize
```

#### 3. 自定义输入输出路径

```bash
python lung_segmentation.py --input_dir /path/to/input --output_dir /path/to/output
```

### 肺叶分析 (lung_lobe_analysis.py)

#### 1. 测试单个病例

```bash
# 测试单个病例（例如 CT_NSIP1）
python lung_lobe_analysis.py --test_single CT_NSIP1
```

#### 2. 批量分析所有病例

```bash
# 分析所有匹配的病例
python lung_lobe_analysis.py
```

#### 3. 自定义参数

```bash
# 自定义输入输出路径和分割百分位数
python lung_lobe_analysis.py --lung_mask_dir /path/to/lung_masks --lesion_dir /path/to/lesions --output_dir /path/to/output --percentile 65
```

## 参数说明

### lung_segmentation.py 参数

- `--input_dir`: 输入CT图像目录（默认：`/home/huchengpeng/workspace/dataset/images`）
- `--output_dir`: 输出结果目录（默认：`/home/huchengpeng/workspace/Lung_Segmentation/results`）
- `--visualize`: 显示可视化结果
- `--test_single`: 测试单个文件，例如：`CT_NSIP1_0000.nii.gz`

### lung_lobe_analysis.py 参数

- `--lung_mask_dir`: 肺部掩码文件目录（默认：`/home/huchengpeng/workspace/Lung_Segmentation/results`）
- `--lesion_dir`: 病变分割文件目录（默认：`/home/huchengpeng/workspace/dataset/labels`）
- `--output_dir`: 输出结果目录（默认：`/home/huchengpeng/workspace/Lung_Segmentation/analysis_results`）
- `--test_single`: 测试单个病例，例如：`CT_NSIP1`
- `--percentile`: 上下肺划分的百分位数（默认：60%）

## 输入文件格式

程序会自动查找以下命名格式的CT文件：
- `CT_UIP{1-23}_0000.nii.gz`
- `CT_NSIP{1-19}_0000.nii.gz`

## 输出文件

### 肺部分割输出

对于每个输入文件 `CT_{name}_0000.nii.gz`，程序会生成：
- `{name}_lung_mask.nii.gz`: 肺部掩码
- `{name}_lung_segmented.nii.gz`: 分割后的CT图像
- `{name}_visualization.png`: 可视化结果（如果启用）

### 肺叶分析输出

- `lung_lobe_results_{timestamp}.jsonl`: 分析结果文件，包含每个病例的：
  - 上下肺体积信息
  - 病变在上下肺的分布比例
  - 病变占比统计
  - 详细的像素和体积计算结果

## 算法原理

### 肺部分割算法

1. **二值化**: 使用阈值（默认-300 HU）进行二值化
2. **边界清理**: 清除边界上的小斑点
3. **连通区域分析**: 保留两个最大的连通区域（左右肺）
4. **形态学操作**: 腐蚀和闭运算优化分割结果
5. **孔洞填充**: 填充肺部内部的孔洞

### 肺叶分析算法

1. **上下肺划分**: 使用几何百分位数方法（默认60%）划分上下肺
2. **体积计算**: 基于像素间距计算真实体积（立方毫米）
3. **病变分布分析**: 计算病变在上下肺的分布比例
4. **占比统计**: 计算病变在各肺叶的占比百分比

## 使用示例

### 完整工作流程

```bash
# 1. 首先进行肺部分割
# 测试单个文件
python lung_segmentation.py --test_single CT_NSIP1_0000.nii.gz --visualize

# 批量处理所有文件
python lung_segmentation.py

# 2. 然后进行肺叶分析
# 测试单个病例
python lung_lobe_analysis.py --test_single CT_NSIP1

# 批量分析所有病例
python lung_lobe_analysis.py
```

### 输出结果示例

肺叶分析的JSONL输出格式：
```json
{
  "filename": "CT_NSIP1",
  "upper_lesion_ratio": 2.45,
  "lower_lesion_ratio": 8.32,
  "total_lesion_ratio": 5.67
}
```

## 注意事项

- 确保输入目录中存在相应的CT文件
- 程序会自动创建输出目录
- 处理大量文件时可能需要较长时间
- 建议先用单个文件测试分割效果
- 肺叶分析需要先运行肺部分割生成肺部掩码
- 确保肺部掩码和病变分割文件的命名格式匹配
- 肺叶分析程序会自动匹配对应的文件对
