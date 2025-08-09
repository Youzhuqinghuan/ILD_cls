#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ILD分割数据集类
实现支持肺分割驱动边界框生成和背景切片处理的数据集

主要功能：
1. 加载预处理后的图像、标签和肺分割数据
2. 基于肺分割生成边界框，而非依赖Ground Truth
3. 支持背景切片（无病变）的训练
4. 实现平衡采样机制
5. 与SAM2模型兼容的数据格式
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import glob
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging
from sam2.utils.transforms import SAM2Transforms
import cv2

class ILDDataset(Dataset):
    """
    ILD分割数据集类
    支持肺分割驱动的边界框生成和背景切片处理
    """
    
    def __init__(self, 
                 data_root: str,
                 split: str = 'train',
                 bbox_shift: int = 20,
                 background_sample_ratio: float = 0.3,
                 confidence_threshold: float = 0.1,
                 target_resolution: int = 1024,
                 enable_background_training: bool = True,
                 lesion_type_mapping: Dict = None):
        """
        初始化数据集
        
        Args:
            data_root: 数据根目录，包含train/val/test子目录
            split: 数据集划分 ('train', 'val', 'test')
            bbox_shift: 边界框随机扰动范围
            background_sample_ratio: 背景切片采样比例
            confidence_threshold: 病变检测置信度阈值
            target_resolution: 目标分辨率（SAM2输入）
            enable_background_training: 是否启用背景切片训练
        """
        self.data_root = Path(data_root)
        self.split = split
        self.bbox_shift = bbox_shift
        self.background_sample_ratio = background_sample_ratio
        self.confidence_threshold = confidence_threshold
        self.enable_background_training = enable_background_training
        
        # 病变类型映射
        self.lesion_type_mapping = lesion_type_mapping or {
            0: "background",
            1: "GGO",
            2: "reticulation", 
            3: "consolidation",
            4: "honeycombing"
        }
        
        # 设置路径
        self.split_dir = self.data_root / split
        self.img_dir = self.split_dir / 'imgs'
        self.gt_dir = self.split_dir / 'gts'
        self.lung_dir = self.split_dir / 'lungs'
        
        # 验证目录存在
        self._validate_directories()
        
        # SAM2变换
        self._transform = SAM2Transforms(resolution=target_resolution, mask_threshold=0)
        
        # 加载文件列表和元数据
        self.file_list = self._load_file_list()
        self.metadata = self._load_metadata()
        
        # 分析数据集
        self.dataset_stats = self._analyze_dataset()
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"加载{split}数据集: {len(self.file_list)}个切片")
        self.logger.info(f"数据集统计: {self.dataset_stats}")
    
    def _validate_directories(self):
        """验证必要目录存在"""
        required_dirs = [self.img_dir, self.gt_dir, self.lung_dir]
        for dir_path in required_dirs:
            if not dir_path.exists():
                raise FileNotFoundError(f"必要目录不存在: {dir_path}")
    
    def _load_file_list(self) -> List[str]:
        """加载文件列表"""
        # 获取所有图像文件
        img_files = sorted(glob.glob(str(self.img_dir / "*.npy")))
        
        # 验证对应的标签和肺分割文件存在
        valid_files = []
        for img_file in img_files:
            basename = os.path.basename(img_file)
            gt_file = self.gt_dir / basename
            lung_file = self.lung_dir / basename
            
            if gt_file.exists() and lung_file.exists():
                valid_files.append(basename.replace('.npy', ''))
            else:
                self.logger.warning(f"缺少对应文件: {basename}")
        
        return valid_files
    
    def _load_metadata(self) -> Dict:
        """加载切片元数据"""
        metadata_file = self.data_root / 'slice_metadata.json'
        if metadata_file.exists():
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata_list = json.load(f)
            
            # 转换为字典格式，便于查找
            metadata_dict = {}
            for item in metadata_list:
                if isinstance(item, dict) and 'patient_id' in item and 'slice_name' in item:
                    slice_id = f"{item['patient_id']}_{item['slice_name']}"
                    metadata_dict[slice_id] = item
            
            return metadata_dict
        else:
            self.logger.warning("未找到元数据文件，使用默认值")
            return {}
    
    def _analyze_dataset(self) -> Dict:
        """分析数据集，统计背景和病变切片"""
        background_count = 0
        lesion_count = 0
        lesion_type_count = {}
        
        for file_id in self.file_list:
            metadata = self.metadata.get(file_id, {})
            primary_lesion = metadata.get('primary_lesion', 0)
            
            # 根据primary_lesion判断是否为背景
            if primary_lesion == 0:
                background_count += 1
            else:
                lesion_count += 1
            
            # 统计所有类型（包括背景）
            lesion_name = metadata.get('primary_lesion_name', 
                                     self.lesion_type_mapping.get(primary_lesion, f'未知类型{primary_lesion}'))
            lesion_type_count[lesion_name] = lesion_type_count.get(lesion_name, 0) + 1
        
        return {
            'total': len(self.file_list),
            'background': background_count,
            'lesion': lesion_count,
            'lesion_types': lesion_type_count
        }
    
    def _get_lung_bbox(self, lung_mask: np.ndarray, padding: int = None) -> np.ndarray:
        """
        基于肺分割生成边界框
        
        Args:
            lung_mask: 肺分割掩码 (256, 256)
            padding: 边界框扩展像素数
            
        Returns:
            边界框坐标 [x_min, y_min, x_max, y_max]，已缩放到1024分辨率
        """
        if padding is None:
            padding = self.bbox_shift
        
        h, w = lung_mask.shape
        
        # 如果没有肺区域，返回整个图像的边界框
        if not np.any(lung_mask):
            return np.array([0, 0, w, h]) * 4  # 缩放到1024
        
        # 找到肺区域的边界
        coords = np.where(lung_mask > 0)
        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()
        
        # 添加随机扰动（仅在训练时）
        if self.split == 'train' and padding > 0:
            x_min = max(0, x_min - random.randint(0, padding))
            x_max = min(w, x_max + random.randint(0, padding))
            y_min = max(0, y_min - random.randint(0, padding))
            y_max = min(h, y_max + random.randint(0, padding))
        else:
            # 验证和测试时使用固定padding
            x_min = max(0, x_min - padding // 2)
            x_max = min(w, x_max + padding // 2)
            y_min = max(0, y_min - padding // 2)
            y_max = min(h, y_max + padding // 2)
        
        # 缩放到1024分辨率
        bbox = np.array([x_min, y_min, x_max, y_max]) * 4
        
        return bbox
    
    def _process_labels(self, gt: np.ndarray, metadata: Dict) -> Tuple[np.ndarray, bool]:
        """
        处理标签，支持背景切片和病变切片
        
        Args:
            gt: 原始标签 (256, 256)
            metadata: 切片元数据
            
        Returns:
            处理后的二值标签 (256, 256) 和是否为背景切片的标志
        """
        has_lesion = metadata.get('has_lesion', True)
        
        # 如果是背景切片
        if not has_lesion:
            return np.zeros_like(gt, dtype=np.uint8), True
        
        # 如果是病变切片，随机选择一个病变类型
        lesion_types = metadata.get('lesion_types', [])
        if not lesion_types:
            # 如果元数据中没有病变类型信息，从标签中提取
            unique_labels = np.unique(gt)
            lesion_types = unique_labels[unique_labels > 0].tolist()
        
        if lesion_types:
            # 随机选择一个病变类型进行训练
            selected_lesion = random.choice(lesion_types)
            gt2d = (gt == selected_lesion).astype(np.uint8)
        else:
            # 如果没有病变，返回全零标签
            gt2d = np.zeros_like(gt, dtype=np.uint8)
            return gt2d, True
        
        return gt2d, False
    
    def __len__(self) -> int:
        return len(self.file_list)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str, torch.Tensor]:
        """
        获取单个数据样本
        
        Returns:
            img_1024: 图像张量 [3, 1024, 1024]
            gt2d: 标签张量 [1, 256, 256]
            bbox: 边界框张量 [4]
            filename: 文件名
            is_background: 是否为背景切片的标志 [1]
        """
        file_id = self.file_list[index]
        
        # 加载数据
        img = np.load(self.img_dir / f"{file_id}.npy")  # (256, 256, 3)
        gt = np.load(self.gt_dir / f"{file_id}.npy")    # (256, 256)
        lung_mask = np.load(self.lung_dir / f"{file_id}.npy")  # (256, 256)
        
        # 获取元数据
        metadata = self.metadata.get(file_id, {})
        
        # 处理图像：转换为SAM2格式
        img_1024 = self._transform(img.copy())  # [3, 1024, 1024]
        
        # 处理标签
        gt2d, is_background = self._process_labels(gt, metadata)
        
        # 生成基于肺分割的边界框
        bbox = self._get_lung_bbox(lung_mask)
        
        # 转换为张量
        gt2d_tensor = torch.tensor(gt2d[None, :, :]).long()  # [1, 256, 256]
        bbox_tensor = torch.tensor(bbox).float()  # [4]
        is_background_tensor = torch.tensor([1.0 if is_background else 0.0]).float()  # [1]
        
        return img_1024, gt2d_tensor, bbox_tensor, file_id, is_background_tensor
    
    def create_balanced_sampler(self) -> Optional[WeightedRandomSampler]:
        """
        创建平衡采样器，控制背景和病变切片的采样比例
        
        Returns:
            WeightedRandomSampler或None（如果不需要平衡采样）
        """
        if not self.enable_background_training or self.split != 'train':
            return None
        
        weights = []
        background_count = self.dataset_stats['background']
        lesion_count = self.dataset_stats['lesion']
        
        if background_count == 0 or lesion_count == 0:
            return None
        
        # 计算权重，使背景切片占指定比例
        background_weight = self.background_sample_ratio / background_count
        lesion_weight = (1 - self.background_sample_ratio) / lesion_count
        
        for file_id in self.file_list:
            metadata = self.metadata.get(file_id, {})
            has_lesion = metadata.get('has_lesion', True)
            
            if has_lesion:
                weights.append(lesion_weight)
            else:
                weights.append(background_weight)
        
        return WeightedRandomSampler(weights, len(weights), replacement=True)
    
    def get_statistics(self) -> Dict:
        """获取数据集统计信息"""
        return {
            'dataset_stats': self.dataset_stats,
            'total_files': len(self.file_list),
            'split': self.split,
            'background_ratio': self.dataset_stats['background'] / self.dataset_stats['total'] if self.dataset_stats['total'] > 0 else 0
        }

def create_dataloader(data_root: str,
                     split: str = 'train',
                     batch_size: int = 8,
                     num_workers: int = 4,
                     bbox_shift: int = 20,
                     background_sample_ratio: float = 0.3,
                     enable_background_training: bool = True,
                     lesion_type_mapping: Dict = None,
                     **kwargs) -> DataLoader:
    """
    创建数据加载器
    
    Args:
        data_root: 数据根目录
        split: 数据集划分
        batch_size: 批次大小
        num_workers: 工作进程数
        bbox_shift: 边界框扰动范围
        background_sample_ratio: 背景切片采样比例
        enable_background_training: 是否启用背景切片训练
        **kwargs: 其他参数
        
    Returns:
        DataLoader实例
    """
    dataset = ILDDataset(
        data_root=data_root,
        split=split,
        bbox_shift=bbox_shift,
        background_sample_ratio=background_sample_ratio,
        enable_background_training=enable_background_training,
        lesion_type_mapping=lesion_type_mapping,
        **kwargs
    )
    
    # 创建采样器
    sampler = dataset.create_balanced_sampler()
    shuffle = (split == 'train') and (sampler is None)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train')
    )
    
    return dataloader

# 测试代码
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import torchvision.transforms as transforms
    
    def test_dataset():
        """测试数据集功能"""
        print("开始测试ILD数据集...")
        
        # 设置数据路径（需要根据实际情况修改）
        data_root = "data/preprocessed"
        
        try:
            # 测试训练集
            print("\n=== 测试训练集 ===")
            train_dataset = ILDDataset(
                data_root=data_root,
                split='train',
                bbox_shift=20,
                background_sample_ratio=0.3,
                enable_background_training=True
            )
            
            print(f"训练集大小: {len(train_dataset)}")
            print(f"数据集统计: {train_dataset.get_statistics()}")
            
            # 测试数据加载
            print("\n=== 测试数据加载 ===")
            for i in range(min(3, len(train_dataset))):
                img, gt, bbox, filename, is_background = train_dataset[i]
                print(f"样本 {i}:")
                print(f"  文件名: {filename}")
                print(f"  图像形状: {img.shape}")
                print(f"  标签形状: {gt.shape}")
                print(f"  边界框: {bbox}")
                print(f"  是否背景: {is_background.item() > 0.5}")
                
                # 验证数据范围
                assert img.shape == (3, 1024, 1024), f"图像形状错误: {img.shape}"
                assert gt.shape == (1, 256, 256), f"标签形状错误: {gt.shape}"
                assert bbox.shape == (4,), f"边界框形状错误: {bbox.shape}"
                assert is_background.shape == (1,), f"背景标志形状错误: {is_background.shape}"
                assert torch.all(gt >= 0) and torch.all(gt <= 1), f"标签值范围错误: {gt.min()}-{gt.max()}"
                
                print("  ✓ 数据格式验证通过")
            
            # 测试DataLoader
            print("\n=== 测试DataLoader ===")
            train_loader = create_dataloader(
                data_root=data_root,
                split='train',
                batch_size=4,
                num_workers=0,  # 测试时使用0避免多进程问题
                background_sample_ratio=0.3
            )
            
            batch_count = 0
            for batch_imgs, batch_gts, batch_bboxes, batch_names, batch_is_background in train_loader:
                print(f"批次 {batch_count}:")
                print(f"  批次大小: {len(batch_names)}")
                print(f"  图像批次形状: {batch_imgs.shape}")
                print(f"  标签批次形状: {batch_gts.shape}")
                print(f"  边界框批次形状: {batch_bboxes.shape}")
                print(f"  背景标志形状: {batch_is_background.shape}")
                
                # 统计背景切片比例
                background_count = torch.sum(batch_is_background > 0.5).item()
                print(f"  背景切片数量: {background_count}/{len(batch_names)}")
                
                batch_count += 1
                if batch_count >= 3:  # 只测试前3个批次
                    break
            
            print("\n=== 测试验证集 ===")
            val_dataset = ILDDataset(
                data_root=data_root,
                split='val',
                enable_background_training=False  # 验证时不启用背景训练
            )
            print(f"验证集大小: {len(val_dataset)}")
            
            # 测试边界框生成
            print("\n=== 测试边界框生成 ===")
            sample_img, sample_gt, sample_bbox, sample_name, sample_is_background = val_dataset[0]
            print(f"样本边界框: {sample_bbox}")
            print(f"边界框范围检查: x=[{sample_bbox[0]:.1f}, {sample_bbox[2]:.1f}], y=[{sample_bbox[1]:.1f}, {sample_bbox[3]:.1f}]")
            print(f"是否背景切片: {sample_is_background.item() > 0.5}")
            
            # 验证边界框合理性
            assert sample_bbox[0] >= 0 and sample_bbox[2] <= 1024, "X坐标超出范围"
            assert sample_bbox[1] >= 0 and sample_bbox[3] <= 1024, "Y坐标超出范围"
            assert sample_bbox[0] < sample_bbox[2], "X坐标顺序错误"
            assert sample_bbox[1] < sample_bbox[3], "Y坐标顺序错误"
            print("  ✓ 边界框验证通过")
            
            print("\n🎉 所有测试通过！")
            
        except FileNotFoundError as e:
            print(f"❌ 文件未找到错误: {e}")
            print("请确保数据预处理已完成，并且数据路径正确")
        except Exception as e:
            print(f"❌ 测试失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 运行测试
    test_dataset()