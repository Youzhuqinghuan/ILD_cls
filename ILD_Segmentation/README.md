# ILD Segmentation Module - MedSAM2 Fine-tuning Pipeline

This module implements the **ILD (Interstitial Lung Disease) segmentation system** based on MedSAM2, specifically designed for 4-class lesion segmentation in HRCT images. The pipeline converts raw NIfTI medical images through complete preprocessing, training, inference, and evaluation workflows.

## 核心架构概述

### 完整训练工作流程
```
Raw Data (NIfTI) → Preprocessing → Training → Inference → Evaluation
     ↓                   ↓            ↓          ↓            ↓
[原始CT图像]          [数据格式转换]    [微调SAM2]   [3D分割推理]   [Dice评估]
  ├─ CT_*.nii.gz       ├─ nii→npz      ├─ 冻结提示编码器  ├─ slice-by-slice  ├─ 多类别DSC
  └─ Labels_*.nii.gz   └─ npz→npy      └─ 训练图像编码器  └─ 3D体积重建      └─ 可视化结果
```

## 训练入口：train.sh 脚本分析

### 执行流程概览
`train.sh` 是整个训练系统的统一入口，集成了数据预处理、模型训练、推理和评估四个阶段：

```bash
# 核心执行阶段
1. 数据预处理 (nii_to_npz.py)    # NIfTI → NPZ 格式转换
2. 训练数据准备 (npz_to_npy.py)   # NPZ → NPY 格式转换  
3. 模型训练 (finetune_sam2_img.py) # SAM2微调训练
4. 推理评估 (infer_medsam2_ILD.py + compute_metrics_ILD_all.py)
```

### 关键配置参数
```bash
# 数据路径配置
IMG_PATH="/home/huchengpeng/ILD/imagesTr"          # CT图像路径
GT_PATH="/home/huchengpeng/ILD/labelsTr"           # 标签路径
IMG_NAME_SUFFIX="_0000.nii.gz"                     # CT文件后缀
GT_NAME_SUFFIX=".nii.gz"                           # 标签文件后缀

# 预处理参数 (CT窗宽窗位设置)
WINDOW_LEVEL=40                                     # 软组织窗位
WINDOW_WIDTH=400                                    # 软组织窗宽

# 训练超参数
EPOCHS=500                                          # 训练轮数
BATCH_SIZE=16                                       # 批处理大小
L_RATE=1e-5                                        # 学习率
MODEL_CFG="sam2_hiera_t.yaml"                      # 模型配置(Tiny版本)
```

## 数据预处理管道详细分析

### 数据流概览：从患者级别到切片级别

```
患者级别 CT 文件 → ROI 提取 → 切片分解 → 训练/推理
       ↓              ↓           ↓           ↓
[3D NIfTI 全体积]  [NPZ 有效切片]  [NPY 单切片]  [边界框指导]
  ├─ CT_NSIP1_0000.nii.gz  ├─ 仅病灶切片    ├─ 单独保存     ├─ 训练时：GT计算
  ├─ (512, 512, 200)       ├─ 空间压缩     ├─ 256×256      ├─ 推理时：传播计算
  └─ 完整胸部扫描           └─ ROI提取      └─ 3通道格式     └─ 3D连续性保持
```

### 阶段1: NIfTI → NPZ 转换 (nii_to_npz.py)

#### 核心功能：患者级别数据读取与ROI提取
- **输入**: 原始CT图像 (`CT_*_0000.nii.gz`) 和对应标签 (`CT_*.nii.gz`)
- **输出**: 预处理后的NPZ压缩文件，包含图像、标签和空间信息
- **处理级别**: 以患者为单位，处理完整3D体积

#### 详细数据读取流程

1. **患者文件配对读取**
```python
def preprocess(name, npz_path):
    # 通过文件名自动配对CT图像和标签
    image_name = name.split(gt_name_suffix)[0] + img_name_suffix
    # 例：CT_NSIP1.nii.gz → CT_NSIP1_0000.nii.gz
    
    # 先读取标签确定ROI位置
    gt_sitk = sitk.ReadImage(join(gt_path, gt_name))
    gt_data_ori = np.uint8(sitk.GetArrayFromImage(gt_sitk))  # Shape: (Z, H, W)
    
    # 再读取对应的CT图像
    img_sitk = sitk.ReadImage(join(nii_path, image_name))
    image_data = sitk.GetArrayFromImage(img_sitk)  # Shape: (Z, H, W)
```

2. **智能ROI层面选择策略**
```python
# 步骤1: 3D小目标过滤 (去除<1000像素的3D连通域)
gt_data_ori = cc3d.dust(gt_data_ori, threshold=1000, connectivity=26)

# 步骤2: 2D小目标过滤 (每个切片去除<100像素的区域)
for slice_i in range(gt_data_ori.shape[0]):
    gt_i = gt_data_ori[slice_i, :, :]
    gt_data_ori[slice_i, :, :] = cc3d.dust(gt_i, threshold=100, connectivity=8)

# 步骤3: 识别所有包含病灶的Z轴切片
z_index, _, _ = np.where(gt_data_ori > 0)
z_index = np.unique(z_index)

# 步骤4: 仅提取有效ROI切片 (大幅减少存储空间)
if len(z_index) > 0:
    gt_roi = gt_data_ori[z_index, :, :]      # 仅病灶切片
    img_roi = image_data_pre[z_index, :, :]  # 对应CT切片
    
# 结果：从~200层减少到~20-50层有效切片
```

3. **ROI选择的关键优势**
- **存储优化**: 平均减少60-80%的存储空间
- **计算效率**: 训练时无需处理空白切片  
- **质量提升**: 小目标过滤去除标注噪声
- **上下文保持**: 保留病灶周围的空间信息

4. **CT图像窗宽窗位标准化**
```python
# 软组织窗设置 (Window Level=40, Window Width=400)
lower_bound = WINDOW_LEVEL - WINDOW_WIDTH / 2  # -160 HU
upper_bound = WINDOW_LEVEL + WINDOW_WIDTH / 2  # 240 HU
image_data_pre = np.clip(image_data, lower_bound, upper_bound)

# 归一化到 [0, 255] 确保一致的对比度
image_data_pre = ((image_data_pre - np.min(image_data_pre)) / 
                  (np.max(image_data_pre) - np.min(image_data_pre)) * 255.0)
image_data_pre = np.uint8(image_data_pre)
```

### 阶段2: NPZ → NPY 转换 (npz_to_npy.py)

#### 核心功能：切片级别数据准备
将NPZ格式转换为训练友好的NPY格式，实现从患者级别到切片级别的转换

#### 数据单位转换策略

1. **3D患者 → 2D切片分解**
```python
def convert_npz_to_npy(npz_name, npz_dir, npy_dir):
    npz = np.load(npz_path, allow_pickle=True, mmap_mode="r")
    imgs = npz["imgs"]  # Shape: (N_slices, H, W) - 患者的有效切片
    gts = npz["gts"]    # Shape: (N_slices, H, W) - 对应标签
    
    # 逐slice分解并独立保存
    for i in range(imgs.shape[0]):
        img_i = imgs[i, :, :]  # 单个切片
        gt_i = gts[i, :, :]    # 单个标签
        
        # 切片文件命名: 患者名-切片索引
        slice_name = name + "-" + str(i).zfill(3)  # 例: CT_NSIP1-001.npy
        np.save(join(npy_dir, "imgs", slice_name + ".npy"), img_3c)
        np.save(join(npy_dir, "gts", slice_name + ".npy"), gt_i)
```

2. **训练优化的格式标准化**
```python
# 图像格式适配SAM2输入要求
img_3c = np.repeat(img_i[:, :, None], 3, axis=-1)  # 灰度→RGB 3通道

# 标签分辨率优化 (提高训练效率)  
gt_i = cv2.resize(gt_i, (256, 256), interpolation=cv2.INTER_NEAREST)
# 原始: (512, 512) → 训练: (256, 256)，推理时再upscale到1024
```

#### 数据单位对比总结

| 阶段 | 数据单位 | 文件格式 | 典型尺寸 | 用途 |
|------|----------|----------|----------|------|
| 预处理输入 | 患者级别 | .nii.gz | (200, 512, 512) | 完整CT扫描 |
| NPZ中间格式 | 患者级别 | .npz | (50, 512, 512) | ROI提取后 |
| NPY训练格式 | **切片级别** | .npy | (256, 256) | 单切片训练 |
| 推理处理 | 患者级别→切片级别 | .npz→逐slice | (1024, 1024) | 逐slice推理 |

## 训练与推理的数据处理策略

### 训练时：切片级别处理 + 边界框计算

#### NpyDataset 数据加载机制
```python
class NpyDataset(Dataset):
    def __getitem__(self, index):
        # 1. 加载单个切片数据
        img = np.load(img_path)    # Shape: (1024, 1024, 3) 
        gt = np.load(gt_path)      # Shape: (256, 256), 多类别标签
        
        # 2. 随机选择单个病灶类别 (数据平衡策略)
        label_ids = np.unique(gt)[1:]  # 排除背景(0)
        selected_label = random.choice(label_ids.tolist())
        gt2D = np.uint8(gt == selected_label)  # 转为二值mask
        
        # 3. 基于GT标签计算紧致边界框
        y_indices, x_indices = np.where(gt2D > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices) 
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        
        # 4. 边界框随机扰动 (数据增强)
        x_min = max(0, x_min - random.randint(0, self.bbox_shift))  # bbox_shift=5
        x_max = min(W, x_max + random.randint(0, self.bbox_shift))
        y_min = max(0, y_min - random.randint(0, self.bbox_shift)) 
        y_max = min(H, y_max + random.randint(0, self.bbox_shift))
        
        # 5. 坐标尺度适配 (256 → 1024)
        bboxes = np.array([x_min, y_min, x_max, y_max]) * 4
        
        return img_1024, gt2D, bboxes, img_name
```

#### 训练时边界框确定原理
- **数据来源**: 直接从Ground Truth标签计算
- **计算方法**: 最小外接矩形 + 随机扰动  
- **扰动范围**: ±5像素 (提高模型鲁棒性)
- **尺度转换**: 标签256×256 → 模型输入1024×1024

### 推理时：患者级别处理 + 3D边界框传播

#### 推理数据单位与边界框策略
```python
def main(name):  # name: 患者级别NPZ文件
    npz = np.load(npz_path)
    img_3D = npz['imgs']   # 患者完整有效切片 (N, H, W)
    gt_3D = npz['gts']     # 用于确定初始边界框 (测试时有GT)
    
    # 对每个病灶类别分别进行3D分割
    for label_id in label_ids:
        # 1. 初始边界框计算：找最大病灶切片
        for z in marker_zids:  # 包含该类别病灶的所有切片
            z_box = get_bbox(marker_data_id[z, :, :])  # 每层的边界框
            bbox_dict[z] = z_box
        
        # 2. 确定起始切片：选择病灶面积最大的切片
        bbox_areas = [np.prod(bbox_dict[z][2:] - bbox_dict[z][:2]) for z in bbox_dict.keys()]
        z_max_area = list(bbox_dict.keys())[np.argmax(bbox_areas)]
        z_middle = int((z_max - z_min)/2 + z_min)
        
        # 3. 双向传播推理策略
        # 向上推理 (z_middle → z_max)
        for z in range(z_middle, z_max):
            if z == z_middle:
                # 起始切片：使用最大病灶的边界框
                box_1024 = mid_slice_bbox_2d / np.array([W, H, W, H]) * 1024
            else:
                # 后续切片：基于前一切片的分割结果计算边界框  
                pre_seg = segs_3d_temp[z-1, :, :]
                if np.max(pre_seg) > 0:
                    box_1024 = get_bbox(pre_seg)  # 传播边界框
                else:
                    box_1024 = mid_slice_bbox_2d / np.array([W, H, W, H]) * 1024  # 回退策略
            
            # 单切片推理
            img_2d_seg = medsam_inference(medsam_model, features, box_1024, H, W)
            segs_3d_temp[z, img_2d_seg>0] = 1
        
        # 向下推理 (z_middle-1 → z_min) - 同样的传播策略
```

#### 推理时边界框确定的3D策略

1. **初始定位**: 
   - 分析每个切片的病灶分布
   - 选择病灶面积最大的切片作为起始点
   - 确保从最可靠的位置开始推理

2. **3D传播机制**:
   - **前向传播**: 使用前一切片的分割结果指导当前切片
   - **自适应调整**: 若前一切片无分割结果，回退到初始边界框
   - **连续性保持**: 确保3D体积的空间连续性

3. **边界框传播优势**:
   - **上下文利用**: 充分利用3D空间信息
   - **误差纠正**: 单切片错误不会无限传播  
   - **效率提升**: 避免全图搜索，聚焦ROI区域

### 数据处理流程总结

#### 完整数据流对比

| 处理阶段 | 输入数据 | 输出数据 | 数据单位 | 边界框来源 | 主要目的 |
|----------|----------|----------|----------|------------|----------|
| **预处理** | NIfTI完整CT | NPZ ROI切片 | 患者级别 | - | 存储优化+质量提升 |
| **训练准备** | NPZ患者数据 | NPY单切片 | 切片级别 | GT标签计算 | 训练数据准备 |
| **模型训练** | NPY单切片 | 分割预测 | 切片级别 | GT+随机扰动 | 模型学习 |
| **模型推理** | NPZ患者数据 | 3D分割结果 | 患者→切片 | 3D传播策略 | 临床应用 |

#### 关键技术创新

1. **自适应ROI提取**: 基于病灶分布智能选择有效切片
2. **多尺度数据流**: 预处理(512)→训练(256)→推理(1024)的灵活尺度转换  
3. **3D-2D协同**: 保持3D上下文的逐切片处理策略
4. **智能边界框传播**: 从可靠起点开始的双向3D传播机制

## 模型训练详细实现 (finetune_sam2_img.py)

### SAM2架构适配

#### MedSAM2模型定义
```python
class MedSAM2(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.sam2_model = model
        # 🔒 冻结提示编码器 (专注于图像理解)
        for param in self.sam2_model.sam_prompt_encoder.parameters():
            param.requires_grad = False
```

#### 可训练参数统计
```python
# 仅训练图像编码器 + 掩膜解码器
img_mask_encdec_params = (
    list(medsam_model.sam2_model.image_encoder.parameters()) + 
    list(medsam_model.sam2_model.sam_mask_decoder.parameters())
)
```

### 数据增强策略

#### NpyDataset类实现
```python
class NpyDataset(Dataset):
    def __getitem__(self, index):
        # 随机选择病灶类别 (数据平衡)
        label_ids = np.unique(gt)[1:]  # 排除背景
        gt2D = np.uint8(gt == random.choice(label_ids.tolist()))
        
        # 边界框扰动 (提高鲁棒性)
        x_min = max(0, x_min - random.randint(0, self.bbox_shift))
        x_max = min(W, x_max + random.randint(0, self.bbox_shift))
        
        # 坐标缩放 256→1024
        bboxes = np.array([x_min, y_min, x_max, y_max]) * 4
```

### 训练循环核心逻辑

#### 损失函数组合
```python
# Dice Loss + 交叉熵 (处理类别不平衡)
seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
ce_loss = nn.BCEWithLogitsLoss(reduction="mean")
total_loss = seg_loss(medsam_pred, gt2D) + ce_loss(medsam_pred, gt2D.float())
```

#### 模型保存策略
```python
# 实时保存最新检查点
torch.save(checkpoint, join(model_save_path, "medsam_model_latest.pth"))
# 基于验证损失保存最佳模型
if val_loss < best_loss:
    torch.save(checkpoint, join(model_save_path, "medsam_model_best.pth"))
```

## 推理系统实现 (infer_medsam2_ILD.py)

### 3D推理策略

#### 标签映射定义
```python
label_dict = {
    1: 'ILD1',           # 未知类型1
    2: 'Reticulation',   # 网状影  
    3: 'GGO-reticulation', # 磨玻璃-网状混合影
    4: 'ILD4',           # 未知类型4
    5: 'Honeycombing',   # 蜂窝状
}
```

#### Slice-by-Slice 推理流程
1. **中心切片定位**: 找到病灶最大的切片作为起始点
2. **双向传播**: 从中心向两端进行序列推理
3. **边界框传播**: 使用前一切片的结果指导当前切片的边界框

```python
def main(name):
    # 对每个病灶类别单独处理
    for label_id in label_ids:
        # 找到病灶最大的切片
        z_max_area = list(bbox_dict.keys())[np.argmax(bbox_areas)]
        z_middle = int((z_max - z_min)/2 + z_min)
        
        # 向上推理 (z_middle → z_max)
        for z in range(z_middle, z_max):
            if z == z_middle:
                box_1024 = mid_slice_bbox_2d / np.array([W, H, W, H]) * 1024
            else:
                # 使用前一切片的分割结果指导边界框
                pre_seg = segs_3d_temp[z-1, :, :]
                box_1024 = get_bbox(pre_seg1024)
        
        # 向下推理 (z_middle-1 → z_min)  
        for z in range(z_middle-1, z_min, -1):
            pre_seg = segs_3d_temp[z+1, :, :]
            box_1024 = get_bbox(pre_seg1024)
```

## 评估系统实现 (compute_metrics_ILD_all.py)

### 多类别Dice评估

#### 核心评估指标
```python
def compute_dice_coefficient(mask_gt, mask_pred):
    """计算Soerensen-Dice系数"""
    intersection = np.sum(mask_gt * mask_pred)
    return 2 * intersection / (np.sum(mask_gt) + np.sum(mask_pred))

def compute_multi_class_dsc(gt_data, pred_data):
    """计算多类别DSC"""
    for i in label_dict.keys():
        gt_i = (gt_data == i).astype(np.uint8)
        pred_i = (pred_data == i).astype(np.uint8)
        
        if np.sum(gt_i) == 0 and np.sum(pred_i) == 0:
            dsc[label_dict[i]] = np.nan  # 都为空时返回NaN
        elif np.sum(gt_i) == 0 and np.sum(pred_i) > 0:
            dsc[label_dict[i]] = 0       # GT为空但预测非空时返回0
        else:
            dsc[label_dict[i]] = compute_dice_coefficient(gt_i, pred_i)
```

#### 评估输出
1. **平均DSC计算**: 对每个类别计算所有有效切片的平均Dice系数
2. **CSV结果保存**: 生成详细的评估报告
3. **可视化图表**: 为每个类别生成DSC分布柱状图

## ILD特定的4类分割标签

本系统专门针对ILD的4类病灶进行分割：

| 标签ID | 病灶类型 | 临床意义 |
|--------|----------|----------|
| 1 | ILD1 | 待分类间质病变 |
| 2 | Reticulation | 网状影 (纤维化标志) |
| 3 | GGO-reticulation | 混合型病变 |
| 4 | ILD4 | 待分类间质病变 |
| 5 | Honeycombing | 蜂窝状 (终末期纤维化) |

## 技术特点与创新

### 1. 医学图像适配优化
- **窗宽窗位标准化**: 采用软组织窗 (40/400) 确保一致的对比度
- **3D上下文利用**: 通过slice传播保持3D连续性
- **小目标过滤**: 去除噪声标注提高分割质量

### 2. 训练策略优化  
- **选择性微调**: 仅训练图像编码器和掩膜解码器
- **数据增强**: 边界框扰动 + 随机类别选择
- **损失函数**: Dice+CE组合处理类别不平衡

### 3. 推理效率优化
- **中心启动策略**: 从最大病灶切片开始推理
- **自适应边界框**: 基于前一切片结果指导当前推理
- **多进程并行**: 支持多worker并行推理

## 内存与计算要求

- **训练要求**: ~42GB GPU内存 (A6000, batch_size=16)
- **推理要求**: ~8GB GPU内存 
- **预处理**: 支持多进程并行 (可配置worker数量)
- **存储需求**: NPZ格式比原始NIfTI减少~60%存储空间

## 安装与环境配置

### 环境要求
- `Ubuntu 20.04` | Python `3.10` | `CUDA 12.1+` | `PyTorch 2.3.1`

### 安装步骤
```bash
# 1. 创建虚拟环境
conda create -n sam2_in_med python=3.10 -y
conda activate sam2_in_med

# 2. 安装PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. 克隆MedSAM2仓库
git clone -b MedSAM2 https://github.com/bowang-lab/MedSAM/

# 4. 设置CUDA环境
export CUDA_HOME=/usr/local/cuda-12.1

# 5. 安装MedSAM2
cd MedSAM2 && pip install -e .

# 6. 安装依赖包
pip install SimpleITK nibabel scikit-image connected-components-3d
pip install monai matplotlib pandas tqdm
```

### 模型权重下载
```bash
# 下载SAM2预训练权重
mkdir checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt -O checkpoints/sam2_hiera_tiny.pt
```

## 使用指南

### 完整训练流程
```bash
# 1. 配置train.sh中的数据路径
# 2. 设置训练/推理模式
TRAINORINFER=1  # 训练模式

# 3. 执行完整pipeline
bash train.sh
```

### 仅推理模式
```bash
# 设置推理模式
TRAINORINFER=2  # 推理模式
USEMEDSAM=true  # 使用微调后的MedSAM2

# 执行推理
bash train.sh
```

### 单独执行各阶段
```bash
# 数据预处理
python nii_to_npz.py -img_path /path/to/images -gt_path /path/to/labels -output_path ./data/ILD

# 格式转换
python npz_to_npy.py -npz_train_dir ./data/ILD/npz_train/CT_ILD -npy_dir ./data/ILD/npy

# 模型训练
python finetune_sam2_img.py -i ./data/ILD/npy -task_name MedSAM2-Tiny-ILD

# 推理评估
python infer_medsam2_ILD.py -data_root ./data/ILD/npz_test/CT_ILD -pred_save_dir ./segs/medsam2
python ./metrics/compute_metrics_ILD_all.py -s ./segs/medsam2 -g /path/to/ground_truth
```

## 致谢

- 感谢Meta AI开源SAM2模型
- 感谢MedSAM团队提供医学图像分割基础框架
- 基于MedSAM2论文: [Segment Anything in Medical Images and Videos](https://arxiv.org/abs/2408.03322)

---

**本文档提供了ILD_Segmentation模块的完整技术实现分析，涵盖从数据预处理到模型评估的全部流程。**