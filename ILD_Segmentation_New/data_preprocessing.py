#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ILD分割数据预处理模块
实现改进的数据预处理流程，支持统一配置文件

主要功能：
1. CT、label、lungs文件的读取
2. 基于病变标签分布的患者级数据划分
3. 生成训练用的npy文件

根据ILD_Segmentation_Implementation_Plan.md实现
"""

import os
import json
import yaml
import numpy as np
import SimpleITK as sitk
import cv2
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import logging
from tqdm import tqdm
import argparse
import multiprocessing as mp
from functools import partial
import cc3d
from sklearn.model_selection import train_test_split
from collections import defaultdict, Counter
import sys

class LungMaskLoader:
    """肺分割加载器，用于生成基于肺分割的边界框"""
    
    def __init__(self, lung_mask_dir: str, patterns: List[str]):
        """
        初始化肺分割加载器
        
        Args:
            lung_mask_dir: 肺分割文件目录
            patterns: 文件名模式列表
        """
        self.lung_mask_dir = Path(lung_mask_dir)
        self.patterns = patterns
        self.logger = logging.getLogger(__name__)
        
        if not self.lung_mask_dir.exists():
            self.logger.warning(f"肺分割目录不存在: {lung_mask_dir}")
    
    def load_lung_mask(self, case_name: str) -> Optional[np.ndarray]:
        """
        加载指定病例的肺分割
        
        Args:
            case_name: 病例名称
            
        Returns:
            肺分割数组，如果文件不存在返回None
        """
        for pattern in self.patterns:
            filename = pattern.format(case_name=case_name)
            lung_path = self.lung_mask_dir / filename
            
            if lung_path.exists():
                try:
                    lung_sitk = sitk.ReadImage(str(lung_path))
                    lung_mask = sitk.GetArrayFromImage(lung_sitk)
                    return (lung_mask > 0).astype(np.uint8)
                except Exception as e:
                    self.logger.error(f"读取肺分割文件失败 {lung_path}: {e}")
                    continue
        
        self.logger.warning(f"未找到病例 {case_name} 的肺分割文件")
        return None
    
    def get_lung_bbox_2d(self, lung_mask_2d: np.ndarray, padding: int = 5) -> np.ndarray:
        """
        基于2D肺分割生成边界框
        
        Args:
            lung_mask_2d: 2D肺分割掩码
            padding: 边界框扩展像素数
            
        Returns:
            边界框坐标 [x_min, y_min, x_max, y_max]
        """
        if not np.any(lung_mask_2d):
            h, w = lung_mask_2d.shape
            return np.array([0, 0, w, h])
        
        coords = np.where(lung_mask_2d > 0)
        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()
        
        h, w = lung_mask_2d.shape
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)
        
        return np.array([x_min, y_min, x_max, y_max])

class LesionBasedDataSplitter:
    """基于病变类型的切片级数据划分器"""
    
    def __init__(self, lesion_analysis_file: str, config: Dict, seed: int = 42):
        """
        初始化数据划分器
        
        Args:
            lesion_analysis_file: 病变分析文件路径
            seed: 随机种子
        """
        self.lesion_analysis_file = lesion_analysis_file
        self.seed = seed
        self.logger = logging.getLogger(__name__)
        self.lesion_data = self._load_lesion_data()
        
        # 从配置文件读取病变类型映射
        self.lesion_type_map = config['data']['lesion_type_mapping']
    
    def _load_lesion_data(self) -> Dict:
        """加载病变分析数据"""
        try:
            with open(self.lesion_analysis_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.logger.info(f"加载了 {len(data)} 个患者的病变分析信息")
            return data
        except Exception as e:
            self.logger.error(f"加载病变分析文件失败: {e}")
            return {}
    
    def stratified_lesion_split(self, train_ratio: float, val_ratio: float, test_ratio: float) -> Dict[str, List[str]]:
        """
        基于病变类型的分层切片级划分
        
        Args:
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            
        Returns:
            包含各数据集切片列表的字典
        """
        if not self.lesion_data:
            self.logger.error("没有病变数据，无法进行数据划分")
            return {'train': [], 'val': [], 'test': []}
        
        # 收集所有切片及其病变类型
        slice_lesion_map = {}
        lesion_type_groups = defaultdict(list)
        
        for patient_id, patient_data in self.lesion_data.items():
            slice_lesions = patient_data.get('slice_lesion_labels', {})
            
            for slice_name, lesion_types in slice_lesions.items():
                slice_id = f"{patient_id}_{slice_name}"
                slice_lesion_map[slice_id] = lesion_types
                
                # 按主要病变类型分组（取最大的病变类型ID）
                if lesion_types:
                    primary_lesion = max(lesion_types)
                    lesion_type_groups[primary_lesion].append(slice_id)
        
        # 统计病变类型分布
        lesion_distribution = {}
        for lesion_type, slices in lesion_type_groups.items():
            lesion_name = self.lesion_type_map.get(lesion_type, f'Unknown_{lesion_type}')
            lesion_distribution[lesion_name] = len(slices)
        
        self.logger.info(f"病变类型分布: {lesion_distribution}")
        
        train_slices, val_slices, test_slices = [], [], []
        
        # 对每种病变类型进行分层划分
        for lesion_type, slices in lesion_type_groups.items():
            lesion_name = self.lesion_type_map.get(lesion_type, f'Unknown_{lesion_type}')
            
            if len(slices) < 3:
                train_slices.extend(slices)
                self.logger.warning(f"病变类型 {lesion_name} 只有 {len(slices)} 个切片，全部分配到训练集")
                continue
            
            # 先分出测试集
            train_val, test = train_test_split(
                slices, 
                test_size=test_ratio, 
                random_state=self.seed
            )
            
            # 再从训练验证集中分出验证集
            if len(train_val) >= 2:
                val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
                train, val = train_test_split(
                    train_val, 
                    test_size=val_ratio_adjusted, 
                    random_state=self.seed
                )
            else:
                train, val = train_val, []
            
            train_slices.extend(train)
            val_slices.extend(val)
            test_slices.extend(test)
            
            self.logger.info(f"病变类型 {lesion_name}: 训练={len(train)}, 验证={len(val)}, 测试={len(test)}")
        
        result = {
            'train': train_slices,
            'val': val_slices,
            'test': test_slices
        }
        
        self.logger.info(f"最终划分结果: 训练={len(train_slices)}, 验证={len(val_slices)}, 测试={len(test_slices)}")
        
        return result
    
    def get_slice_metadata(self) -> Dict[str, Dict]:
        """获取切片元数据"""
        slice_metadata = {}
        
        for patient_id, patient_data in self.lesion_data.items():
            slice_lesions = patient_data.get('slice_lesion_labels', {})
            
            for slice_name, lesion_types in slice_lesions.items():
                slice_id = f"{patient_id}_{slice_name}"
                
                # 提取切片索引
                slice_idx = int(slice_name.split('_')[1])
                
                slice_metadata[slice_id] = {
                    'patient_id': patient_id,
                    'slice_name': slice_name,
                    'slice_index': slice_idx,
                    'lesion_types': lesion_types,
                    'primary_lesion': max(lesion_types) if lesion_types else 0,
                    'primary_lesion_name': self.lesion_type_map.get(max(lesion_types), '背景') if lesion_types else '背景',
                    'has_lesion': len(lesion_types) > 0 and max(lesion_types) > 0
                }
        
        return slice_metadata

class ILDDataPreprocessor:
    """ILD数据预处理器"""
    
    def __init__(self, config: Dict):
        """
        初始化数据预处理器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 提取配置
        self.data_config = config['data']
        self.preprocess_config = config['preprocessing']
        self.system_config = config['system']
        
        # 设置路径
        self.ct_dir = Path(self.data_config['raw_data']['ct_dir'])
        self.label_dir = Path(self.data_config['raw_data']['label_dir'])
        self.lung_dir = Path(self.data_config['raw_data']['lung_dir'])
        self.output_dir = Path(self.preprocess_config['output']['save_dir'] if 'save_dir' in self.preprocess_config['output'] else self.data_config['preprocessed']['output_dir'])
        
        # 创建输出目录结构
        self._create_output_dirs()
        
        # 初始化组件
        self.lung_loader = LungMaskLoader(
            self.lung_dir, 
            self.data_config['naming']['lung_patterns']
        )
        
        # 病变类型映射
        self.lesion_type_map = self.data_config['lesion_type_mapping']
        
        self.splitter = LesionBasedDataSplitter(
            self.data_config['raw_data']['lesion_analysis_file'],
            config,
            config['global']['seed']
        )
        
        self.logger.info(f"初始化数据预处理器完成")
    
    def _create_output_dirs(self):
        """创建输出目录结构"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        for split in ['train', 'val', 'test']:
            (self.output_dir / split).mkdir(exist_ok=True)
            (self.output_dir / split / 'imgs').mkdir(exist_ok=True)
            (self.output_dir / split / 'gts').mkdir(exist_ok=True)
            (self.output_dir / split / 'lungs').mkdir(exist_ok=True)
    
    def _find_files(self, case_name: str) -> Dict[str, Optional[Path]]:
        """查找指定病例的所有相关文件"""
        files = {'ct': None, 'label': None}
        
        # 查找CT文件
        ct_name = case_name + self.data_config['naming']['ct_suffix']
        ct_path = self.ct_dir / ct_name
        if ct_path.exists():
            files['ct'] = ct_path
        
        # 查找标签文件
        label_name = case_name + self.data_config['naming']['label_suffix']
        label_path = self.label_dir / label_name
        if label_path.exists():
            files['label'] = label_path
        
        return files
    
    def _preprocess_ct_image(self, ct_array: np.ndarray) -> np.ndarray:
        """预处理CT图像"""
        ct_config = self.preprocess_config['ct_processing']
        
        # 应用窗位窗宽
        lower_bound = ct_config['window_level'] - ct_config['window_width'] / 2
        upper_bound = ct_config['window_level'] + ct_config['window_width'] / 2
        ct_clipped = np.clip(ct_array, lower_bound, upper_bound)
        
        # 归一化
        norm_range = ct_config['normalize_range']
        ct_normalized = ((ct_clipped - ct_clipped.min()) / 
                        (ct_clipped.max() - ct_clipped.min()) * 
                        (norm_range[1] - norm_range[0]) + norm_range[0])
        
        return ct_normalized.astype(np.uint8)
    
    def _clean_labels(self, label_array: np.ndarray) -> np.ndarray:
        """清理标签数据，移除小的连通组件"""
        label_config = self.preprocess_config['label_processing']
        
        # 3D连通组件清理
        if label_config['clean_3d']['enabled']:
            label_array = cc3d.dust(
                label_array, 
                threshold=label_config['clean_3d']['threshold'], 
                connectivity=label_config['clean_3d']['connectivity'], 
                in_place=False
            )
        
        # 2D连通组件清理
        if label_config['clean_2d']['enabled']:
            for z in range(label_array.shape[0]):
                slice_2d = label_array[z, :, :]
                if np.any(slice_2d):
                    label_array[z, :, :] = cc3d.dust(
                        slice_2d, 
                        threshold=label_config['clean_2d']['threshold'], 
                        connectivity=label_config['clean_2d']['connectivity'], 
                        in_place=False
                    )
        
        return label_array
    

    
    def process_single_slice(self, patient_id: str, slice_name: str, split_name: str, lesion_types: List[int]) -> Dict:
        """
        处理单个切片
        
        Args:
            patient_id: 患者ID
            slice_name: 切片名称
            split_name: 数据集名称
            lesion_types: 病变类型列表
            
        Returns:
            处理结果字典
        """
        try:
            # 查找文件
            files = self._find_files(patient_id)
            if not all(files.values()):
                missing = [k for k, v in files.items() if v is None]
                return {
                    'status': 'error',
                    'message': f'缺少文件: {", ".join(missing)}'
                }
            
            # 读取数据
            ct_sitk = sitk.ReadImage(str(files['ct']))
            label_sitk = sitk.ReadImage(str(files['label']))
            
            ct_array = sitk.GetArrayFromImage(ct_sitk)
            label_array = sitk.GetArrayFromImage(label_sitk)
            
            # 解析切片索引
            try:
                slice_idx = int(slice_name.split('_')[-1])
            except (ValueError, IndexError):
                return {
                    'status': 'error',
                    'message': f'无法解析切片索引: {slice_name}'
                }
            
            # 检查切片索引是否有效
            if slice_idx >= ct_array.shape[0] or slice_idx < 0:
                return {
                    'status': 'error',
                    'message': f'切片索引超出范围: {slice_idx}, 总切片数: {ct_array.shape[0]}'
                }
            
            # 预处理完整的3D数据
            ct_processed = self._preprocess_ct_image(ct_array)
            label_processed = self._clean_labels(label_array)
            
            # 获取切片数据
            ct_slice = ct_processed[slice_idx]
            label_slice = label_processed[slice_idx]
            
            # 加载肺分割（如果可用）
            lung_slice = None
            if self.lung_loader:
                lung_mask = self.lung_loader.load_lung_mask(patient_id)
                if lung_mask is not None and slice_idx < lung_mask.shape[0]:
                    lung_slice = lung_mask[slice_idx]
            
            # 生成文件名
            slice_filename = f"{patient_id}_{slice_name}"
            
            # 处理切片数据
            result = self._process_slice_data(ct_slice, label_slice, lung_slice, slice_filename, split_name)
            
            # 添加病变类型信息
            result['lesion_types'] = lesion_types
            result['primary_lesion'] = max(lesion_types) if lesion_types else 0
            result['has_lesion'] = len(lesion_types) > 0
            
            # 添加主要病变类型名称
            lesion_type_map = self.config['data']['lesion_type_mapping']
            primary_lesion_id = result['primary_lesion']
            result['primary_lesion_name'] = lesion_type_map.get(str(primary_lesion_id), f'未知类型{primary_lesion_id}')
            
            return {
                'status': 'success',
                'slice_data': result
            }
            
        except Exception as e:
            self.logger.error(f"处理切片 {patient_id}_{slice_name} 时发生错误: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def process_single_case(self, case_name: str, split_name: str, lesion_slices: List[int], slice_lesion_labels: Dict[str, List[int]]) -> Dict:
        """处理单个病例，提取指定的病变切片"""
        try:
            # 查找文件
            files = self._find_files(case_name)
            
            if files['ct'] is None:
                return {'case_name': case_name, 'status': 'error', 'message': 'CT文件未找到'}
            
            if files['label'] is None:
                return {'case_name': case_name, 'status': 'error', 'message': '标签文件未找到'}
            
            # 读取完整的CT和标签文件
            ct_sitk = sitk.ReadImage(str(files['ct']))
            ct_array = sitk.GetArrayFromImage(ct_sitk)
            
            label_sitk = sitk.ReadImage(str(files['label']))
            label_array = sitk.GetArrayFromImage(label_sitk).astype(np.uint8)
            
            # 清理标签
            label_cleaned = self._clean_labels(label_array)
            
            # 预处理CT
            ct_processed = self._preprocess_ct_image(ct_array)
            
            # 加载肺分割
            lung_mask = self.lung_loader.load_lung_mask(case_name)
            if lung_mask is None and self.preprocess_config['lung_processing']['fallback_to_hu']:
                # 简单的肺区域检测
                hu_range = self.preprocess_config['lung_processing']['hu_range']
                lung_mask = ((ct_array > hu_range[0]) & (ct_array < hu_range[1])).astype(np.uint8)
            
            processed_slices = []
            
            # 处理每个病变切片
            for slice_idx in lesion_slices:
                if slice_idx >= ct_array.shape[0]:
                    self.logger.warning(f"切片索引 {slice_idx} 超出范围，跳过")
                    continue
                
                # 提取单个切片
                ct_slice = ct_processed[slice_idx]
                label_slice = label_cleaned[slice_idx]
                lung_slice = lung_mask[slice_idx] if lung_mask is not None else None
                
                # 获取该切片的病变标签
                slice_name = f"slice_{slice_idx:03d}"
                lesion_types = slice_lesion_labels.get(slice_name, [])
                
                # 确定主要病变类型
                primary_lesion = max(lesion_types) if lesion_types else 0
                primary_lesion_name = self.lesion_type_map.get(primary_lesion, '背景') if lesion_types else '背景'
                
                # 生成切片文件名
                slice_filename = f"{case_name}_slice_{slice_idx:03d}"
                
                # 处理图像和标签
                processed_slice_data = self._process_slice_data(
                    ct_slice, label_slice, lung_slice, slice_filename, split_name
                )
                
                if processed_slice_data['status'] == 'success':
                    # 添加切片元数据
                    processed_slice_data.update({
                        'case_name': case_name,
                        'patient_id': case_name,
                        'split': split_name,
                        'slice_index': int(slice_idx),
                        'lesion_types': lesion_types.tolist() if hasattr(lesion_types, 'tolist') else list(lesion_types),
                        'primary_lesion': int(primary_lesion) if hasattr(primary_lesion, 'item') else primary_lesion,
                        'primary_lesion_name': primary_lesion_name,
                        'has_lesion': len(lesion_types) > 0 and max(lesion_types) > 0
                    })
                    processed_slices.append(processed_slice_data)
                else:
                    self.logger.warning(f"切片 {slice_filename} 处理失败: {processed_slice_data.get('message', '未知错误')}")
            
            return {
                'case_name': case_name,
                'status': 'success',
                'processed_slices': len(processed_slices),
                'slice_data': processed_slices
            }
            
        except Exception as e:
            self.logger.error(f"处理病例 {case_name} 时出错: {e}")
            return {'case_name': case_name, 'status': 'error', 'message': str(e)}
    
    def _process_slice_data(self, ct_slice: np.ndarray, label_slice: np.ndarray, 
                           lung_slice: Optional[np.ndarray], slice_filename: str, split_name: str) -> Dict:
        """处理单个切片数据"""
        try:
            # 标准化处理完整切片（不进行裁剪）
            output_size = self.preprocess_config['ct_processing']['output_size']
            
            # 调整CT图像大小
            ct_resized = cv2.resize(ct_slice, tuple(output_size), interpolation=cv2.INTER_LINEAR)
            
            # 调整标签大小
            label_resized = cv2.resize(label_slice, tuple(output_size), interpolation=cv2.INTER_NEAREST)
            
            # 处理肺分割掩码
            if lung_slice is not None:
                lung_resized = cv2.resize(lung_slice.astype(np.uint8), tuple(output_size), interpolation=cv2.INTER_NEAREST)
                lung_available = True
            else:
                # 如果没有肺分割，创建全零掩码
                lung_resized = np.zeros(output_size, dtype=np.uint8)
                lung_available = False
            
            # 处理图像为3通道
            if self.preprocess_config['ct_processing']['convert_to_3channel']:
                img_3c = np.repeat(ct_resized[:, :, None], 3, axis=-1)
            else:
                img_3c = ct_resized
            
            # 保存文件
            img_path = self.output_dir / split_name / 'imgs' / f"{slice_filename}.npy"
            gt_path = self.output_dir / split_name / 'gts' / f"{slice_filename}.npy"
            lung_path = self.output_dir / split_name / 'lungs' / f"{slice_filename}.npy"
            
            np.save(img_path, img_3c)
            np.save(gt_path, label_resized)
            np.save(lung_path, lung_resized)
            
            return {
                'status': 'success',
                'original_size': list(ct_slice.shape),
                'final_size': list(output_size),
                'lung_available': lung_available
            }
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def run_preprocessing(self) -> None:
        """运行完整的数据预处理流程"""
        self.logger.info("开始数据预处理流程")
        
        # 1. 基于病变类型的切片级数据划分
        self.logger.info("进行基于病变类型的切片级数据划分...")
        split_config = self.preprocess_config['data_split']
        slice_splits = self.splitter.stratified_lesion_split(
            split_config['train_ratio'],
            split_config['val_ratio'],
            split_config['test_ratio']
        )
        
        # 获取切片元数据
        slice_metadata = self.splitter.get_slice_metadata()
        
        # 保存划分结果
        if self.preprocess_config['output']['save_patient_splits']:
            split_file = self.output_dir / 'slice_splits.json'
            with open(split_file, 'w', encoding='utf-8') as f:
                json.dump(slice_splits, f, ensure_ascii=False, indent=2)
            self.logger.info(f"切片划分结果已保存到: {split_file}")
            
            # 保存切片元数据
            metadata_file = self.output_dir / 'slice_metadata_full.json'
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(slice_metadata, f, ensure_ascii=False, indent=2)
            self.logger.info(f"切片元数据已保存到: {metadata_file}")
        
        # 2. 直接按切片处理数据（不再按患者重新组织）
        all_processed_metadata = []
        
        # 加载病变分析数据
        with open(self.splitter.lesion_analysis_file, 'r', encoding='utf-8') as f:
            lesion_data = json.load(f)
        
        # 调试模式限制切片数量
        if self.system_config['debug']['enabled']:
            max_slices = self.system_config['debug'].get('max_slices', 100)
            for split_name in slice_splits:
                if len(slice_splits[split_name]) > max_slices:
                    slice_splits[split_name] = slice_splits[split_name][:max_slices]
            self.logger.info(f"调试模式：限制每个数据集最多 {max_slices} 个切片")
        
        # 3. 处理各个数据集的切片（按患者分组以优化I/O）
        patient_cache = {}  # 缓存已加载的患者数据
        
        for split_name, slice_list in slice_splits.items():
            if not slice_list:
                self.logger.warning(f"数据集 {split_name} 为空，跳过处理")
                continue
            
            self.logger.info(f"处理 {split_name} 数据集，共 {len(slice_list)} 个切片")
            
            # 按患者分组切片
            patient_slices = defaultdict(list)
            for slice_id in slice_list:
                parts = slice_id.split('_')
                if len(parts) < 3:
                    self.logger.warning(f"无效的slice_id格式: {slice_id}")
                    continue
                patient_id = '_'.join(parts[:2])
                slice_name = '_'.join(parts[2:])
                patient_slices[patient_id].append((slice_id, slice_name))
            
            # 处理每个患者的切片
            for patient_id, slices in tqdm(patient_slices.items(), desc=f"处理{split_name}患者"):
                # 获取患者数据
                patient_data = lesion_data.get(patient_id, {})
                if not patient_data:
                    self.logger.warning(f"找不到患者 {patient_id} 的数据")
                    continue
                
                # 加载并缓存患者的3D数据
                if patient_id not in patient_cache:
                    files = self._find_files(patient_id)
                    if not all(files.values()):
                        self.logger.warning(f"患者 {patient_id} 缺少必要文件")
                        continue
                    
                    try:
                        ct_sitk = sitk.ReadImage(str(files['ct']))
                        label_sitk = sitk.ReadImage(str(files['label']))
                        ct_array = sitk.GetArrayFromImage(ct_sitk)
                        label_array = sitk.GetArrayFromImage(label_sitk)
                        
                        # 预处理完整的3D数据
                        ct_processed = self._preprocess_ct_image(ct_array)
                        label_processed = self._clean_labels(label_array)
                        
                        # 加载肺分割
                        lung_mask = None
                        if self.lung_loader:
                            lung_mask = self.lung_loader.load_lung_mask(patient_id)
                        
                        patient_cache[patient_id] = {
                            'ct': ct_processed,
                            'label': label_processed,
                            'lung': lung_mask
                        }
                    except Exception as e:
                        self.logger.error(f"加载患者 {patient_id} 数据失败: {str(e)}")
                        continue
                
                # 处理该患者的所有切片
                cached_data = patient_cache[patient_id]
                slice_lesion_labels = patient_data.get('slice_lesion_labels', {})
                
                for slice_id, slice_name in slices:
                    try:
                        # 解析切片索引
                        slice_idx = int(slice_name.split('_')[-1])
                        
                        # 检查切片索引是否有效
                        if slice_idx >= cached_data['ct'].shape[0] or slice_idx < 0:
                            self.logger.warning(f"切片索引超出范围: {slice_idx}")
                            continue
                        
                        # 获取切片数据
                        ct_slice = cached_data['ct'][slice_idx]
                        label_slice = cached_data['label'][slice_idx]
                        lung_slice = None
                        if cached_data['lung'] is not None and slice_idx < cached_data['lung'].shape[0]:
                            lung_slice = cached_data['lung'][slice_idx]
                        
                        # 获取病变标签
                        lesion_types = slice_lesion_labels.get(slice_name, [])
                        
                        # 生成文件名并处理切片
                        slice_filename = f"{patient_id}_{slice_name}"
                        result = self._process_slice_data(ct_slice, label_slice, lung_slice, slice_filename, split_name)
                        
                        if result['status'] == 'success':
                            result['lesion_types'] = lesion_types
                            result['primary_lesion'] = max(lesion_types) if lesion_types else 0
                            result['has_lesion'] = len(lesion_types) > 0
                            result['split'] = split_name
                            result['patient_id'] = patient_id
                            result['slice_name'] = slice_name
                            all_processed_metadata.append(result)
                        else:
                            self.logger.warning(f"切片 {slice_id} 处理失败: {result.get('message', '未知错误')}")
                            
                    except Exception as e:
                        self.logger.error(f"处理切片 {slice_id} 时发生错误: {str(e)}")
                        continue
        
        # 4. 保存完整的切片元信息
        if self.preprocess_config['output']['save_slice_metadata']:
            metadata_file = self.output_dir / 'slice_metadata.json'
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(all_processed_metadata, f, ensure_ascii=False, indent=2)
            self.logger.info(f"切片元信息已保存到: {metadata_file}")
        
        # 5. 处理背景切片（如果启用）
        if self.preprocess_config.get('background_slices', {}).get('enabled', False):
            self.logger.info("开始处理背景切片...")
            background_metadata = self._process_background_slices(slice_splits, patient_cache)
            all_processed_metadata.extend(background_metadata)
            
            # 重新保存包含背景切片的元数据
            if self.preprocess_config['output']['save_slice_metadata']:
                metadata_file = self.output_dir / 'slice_metadata.json'
                with open(metadata_file, 'w', encoding='utf-8') as f:
                    json.dump(all_processed_metadata, f, ensure_ascii=False, indent=2)
                self.logger.info(f"更新后的切片元信息已保存到: {metadata_file}")
        
        # 6. 生成最终统计报告（包含背景切片）
        if self.preprocess_config['output']['save_statistics']:
            self._generate_statistics_report(all_processed_metadata)
        
        self.logger.info(f"数据预处理完成！共处理 {len(all_processed_metadata)} 个切片")
        self.logger.info(f"输出目录: {self.output_dir}")
    
    def _identify_background_slices(self, patient_cache: Dict) -> Dict[str, List[Dict]]:
        """识别背景切片并按解剖位置分组
        
        Args:
            patient_cache: 缓存的患者数据
            
        Returns:
            Dict[patient_id, List[slice_info]]: 按患者组织的背景切片信息
        """
        background_config = self.preprocess_config['background_slices']
        selection_config = background_config['selection_strategy']
        
        # 加载病变分析数据
        with open(self.splitter.lesion_analysis_file, 'r', encoding='utf-8') as f:
            lesion_data = json.load(f)
        
        background_slices = defaultdict(list)
        
        for patient_id, cached_data in patient_cache.items():
            patient_lesion_data = lesion_data.get(patient_id, {})
            slice_lesion_labels = patient_lesion_data.get('slice_lesion_labels', {})
            
            ct_shape = cached_data['ct'].shape
            total_slices = ct_shape[0]
            
            # 计算排除边缘切片的范围
            exclude_ratio = selection_config['exclude_edge_ratio']
            start_idx = int(total_slices * exclude_ratio)
            end_idx = int(total_slices * (1 - exclude_ratio))
            
            # 遍历所有切片，找出背景切片
            for slice_idx in range(start_idx, end_idx):
                slice_name = f"slice_{slice_idx:03d}"
                
                # 检查是否已有病变标注
                if slice_name in slice_lesion_labels:
                    continue  # 跳过有病变的切片
                
                # 检查肺组织面积（如果有肺分割）
                lung_area = 0
                if cached_data['lung'] is not None and slice_idx < cached_data['lung'].shape[0]:
                    lung_slice = cached_data['lung'][slice_idx]
                    lung_area = np.sum(lung_slice > 0)
                    
                    if selection_config['prefer_lung_rich'] and lung_area < selection_config['min_lung_area']:
                        continue  # 跳过肺组织面积过小的切片
                
                # 计算解剖位置组（上、中、下肺野）
                position_groups = background_config['anatomical_sampling']['position_groups']
                position_group = int((slice_idx - start_idx) / (end_idx - start_idx) * position_groups)
                position_group = min(position_group, position_groups - 1)  # 确保不超出范围
                
                slice_info = {
                    'slice_idx': slice_idx,
                    'slice_name': slice_name,
                    'position_group': position_group,
                    'lung_area': lung_area
                }
                
                background_slices[patient_id].append(slice_info)
        
        return dict(background_slices)
    
    def _process_background_slices(self, slice_splits: Dict[str, List[str]], patient_cache: Dict) -> List[Dict]:
        """处理背景切片并添加到各数据集
        
        Args:
            slice_splits: 原始切片划分
            patient_cache: 缓存的患者数据
            
        Returns:
            List[Dict]: 处理后的背景切片元数据
        """
        background_config = self.preprocess_config['background_slices']
        ratios = background_config['ratios']
        
        # 识别所有背景切片
        background_slices = self._identify_background_slices(patient_cache)
        
        # 统计各数据集当前的切片数量
        current_counts = {split: len(slices) for split, slices in slice_splits.items()}
        
        # 计算需要添加的背景切片数量
        target_background_counts = {}
        for split, ratio in ratios.items():
            if split in current_counts:
                # 根据比例计算目标背景切片数量
                # 如果当前有N个病变切片，目标比例为R，则背景切片数量为 N * R / (1 - R)
                lesion_count = current_counts[split]
                target_background_count = int(lesion_count * ratio / (1 - ratio))
                target_background_counts[split] = target_background_count
        
        self.logger.info(f"目标背景切片数量: {target_background_counts}")
        
        # 收集所有可用的背景切片
        all_background_candidates = []
        for patient_id, slices in background_slices.items():
            for slice_info in slices:
                slice_info['patient_id'] = patient_id
                all_background_candidates.append(slice_info)
        
        self.logger.info(f"发现 {len(all_background_candidates)} 个候选背景切片")
        
        # 按解剖位置分组
        if background_config['anatomical_sampling']['enabled']:
            position_groups = defaultdict(list)
            for candidate in all_background_candidates:
                position_groups[candidate['position_group']].append(candidate)
            
            # 确保每个位置组有足够的切片
            min_per_group = background_config['anatomical_sampling']['min_slices_per_group']
            for group_id, group_slices in position_groups.items():
                if len(group_slices) < min_per_group:
                    self.logger.warning(f"位置组 {group_id} 只有 {len(group_slices)} 个切片，少于最小要求 {min_per_group}")
        
        # 为各数据集分配背景切片
        processed_background_metadata = []
        
        for split_name, target_count in target_background_counts.items():
            if target_count <= 0:
                continue
            
            # 从候选切片中随机选择
            np.random.seed(self.config['global']['seed'])
            selected_candidates = np.random.choice(
                all_background_candidates, 
                size=min(target_count, len(all_background_candidates)), 
                replace=False
            )
            
            self.logger.info(f"为 {split_name} 数据集选择了 {len(selected_candidates)} 个背景切片")
            
            # 处理选中的背景切片
            for candidate in selected_candidates:
                patient_id = candidate['patient_id']
                slice_idx = candidate['slice_idx']
                slice_name = candidate['slice_name']
                
                if patient_id not in patient_cache:
                    continue
                
                cached_data = patient_cache[patient_id]
                
                # 获取切片数据
                ct_slice = cached_data['ct'][slice_idx]
                # 创建全零标签（背景切片）
                label_slice = np.zeros_like(ct_slice, dtype=np.uint8)
                lung_slice = None
                if cached_data['lung'] is not None and slice_idx < cached_data['lung'].shape[0]:
                    lung_slice = cached_data['lung'][slice_idx]
                
                # 生成文件名并处理切片
                slice_filename = f"{patient_id}_{slice_name}_bg"
                result = self._process_slice_data(ct_slice, label_slice, lung_slice, slice_filename, split_name)
                
                if result['status'] == 'success':
                    result['lesion_types'] = []  # 背景切片无病变类型
                    result['primary_lesion'] = 0  # 背景
                    result['has_lesion'] = False  # 无病变
                    result['split'] = split_name
                    result['patient_id'] = patient_id
                    result['slice_name'] = slice_name
                    result['is_background'] = True  # 标记为背景切片
                    result['position_group'] = candidate['position_group']
                    processed_background_metadata.append(result)
                else:
                    self.logger.warning(f"背景切片 {slice_filename} 处理失败: {result.get('message', '未知错误')}")
        
        self.logger.info(f"成功处理 {len(processed_background_metadata)} 个背景切片")
        return processed_background_metadata

    def _generate_statistics_report(self, metadata: List[Dict]) -> None:
        """生成统计报告"""
        stats = {
            'total_slices': len(metadata),
            'by_split': defaultdict(lambda: {'total_slices': 0, 'lesion_slices': 0, 'background_slices': 0}),
            'by_patient': defaultdict(lambda: {'total_slices': 0, 'lesion_slices': 0, 'background_slices': 0, 'split': ''}),
            'lesion_slices': 0,
            'background_slices': 0,
            'lesion_types': defaultdict(int),
            'primary_lesion_distribution': defaultdict(int)
        }
        
        # 从配置文件读取病变类型映射
        lesion_type_map = self.config['data']['lesion_type_mapping']
        # 病变类型映射：从ID到名称（配置文件中键是字符串格式的数字）
        lesion_type_names = {int(k): v for k, v in lesion_type_map.items()}
        
        for item in metadata:
            # 基本统计
            split_name = item['split']
            patient_id = item['patient_id']
            
            stats['by_split'][split_name]['total_slices'] += 1
            stats['by_patient'][patient_id]['total_slices'] += 1
            stats['by_patient'][patient_id]['split'] = split_name
            
            if item['has_lesion']:
                stats['lesion_slices'] += 1
                stats['by_split'][split_name]['lesion_slices'] += 1
                stats['by_patient'][patient_id]['lesion_slices'] += 1
                
                # 病变类型统计
                for lesion_type in item['lesion_types']:
                    lesion_name = lesion_type_names.get(lesion_type, f'未知类型{lesion_type}')
                    stats['lesion_types'][lesion_name] += 1
            else:
                stats['background_slices'] += 1
                stats['by_split'][split_name]['background_slices'] += 1
                stats['by_patient'][patient_id]['background_slices'] += 1
            
            # 区分原始背景切片和添加的背景切片
            if item.get('is_background', False):
                if 'added_background_slices' not in stats:
                    stats['added_background_slices'] = 0
                    stats['by_split_added_bg'] = defaultdict(int)
                stats['added_background_slices'] += 1
                stats['by_split_added_bg'][split_name] += 1
            
            # 主要病变类型统计
            primary_lesion_name = item.get('primary_lesion_name', '背景')
            stats['primary_lesion_distribution'][primary_lesion_name] += 1
        
        # 转换为普通字典
        stats['by_split'] = dict(stats['by_split'])
        stats['by_patient'] = dict(stats['by_patient'])
        stats['lesion_types'] = dict(stats['lesion_types'])
        stats['primary_lesion_distribution'] = dict(stats['primary_lesion_distribution'])
        if 'by_split_added_bg' in stats:
            stats['by_split_added_bg'] = dict(stats['by_split_added_bg'])
        
        # 保存统计报告
        stats_file = self.output_dir / 'preprocessing_statistics.json'
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        # 打印统计信息
        self.logger.info("=== 数据预处理统计报告 ===")
        self.logger.info(f"总切片数: {stats['total_slices']}")
        self.logger.info(f"病变切片: {stats['lesion_slices']}")
        self.logger.info(f"背景切片: {stats['background_slices']}")
        self.logger.info(f"各数据集分布: {stats['by_split']}")
        self.logger.info(f"病变类型分布: {stats['lesion_types']}")
        self.logger.info(f"主要病变分布: {stats['primary_lesion_distribution']}")
        self.logger.info(f"统计报告已保存到: {stats_file}")

def load_config(config_path: str) -> Dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def setup_logging(log_level: str = "INFO"):
    """设置日志配置"""
    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('preprocessing.log', encoding='utf-8')
        ]
    )

def validate_config(config: Dict) -> bool:
    """验证配置文件"""
    logger = logging.getLogger(__name__)
    
    # 检查必要的路径
    paths_to_check = [
        ('CT目录', config['data']['raw_data']['ct_dir']),
        ('标签目录', config['data']['raw_data']['label_dir']),
        ('标注文件', config['data']['raw_data']['annotations_file']),
        ('病变分析文件', config['data']['raw_data']['lesion_analysis_file'])
    ]
    
    all_valid = True
    for name, path in paths_to_check:
        if not Path(path).exists():
            logger.error(f"{name} 不存在: {path}")
            all_valid = False
        else:
            logger.info(f"{name} 验证通过: {path}")
    
    # 肺分割目录可能不存在
    lung_dir = config['data']['raw_data']['lung_dir']
    if not Path(lung_dir).exists():
        logger.warning(f"肺分割目录不存在: {lung_dir}，将使用HU值阈值生成肺区域")
    else:
        logger.info(f"肺分割目录验证通过: {lung_dir}")
    
    return all_valid

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='ILD分割数据预处理')
    parser.add_argument('--config', type=str, default='config.yaml', 
                       help='配置文件路径 [默认: config.yaml]')
    parser.add_argument('--debug', action='store_true', 
                       help='启用调试模式')
    parser.add_argument('--dry_run', action='store_true',
                       help='只验证配置，不执行预处理')
    
    args = parser.parse_args()
    
    # 加载配置
    try:
        config = load_config(args.config)
        print(f"成功加载配置文件: {args.config}")
    except Exception as e:
        print(f"加载配置文件失败: {e}")
        return 1
    
    # 设置日志
    log_level = config.get('global', {}).get('log_level', 'INFO')
    setup_logging(log_level)
    logger = logging.getLogger(__name__)
    
    # 如果启用调试模式，更新配置
    if args.debug:
        config['system']['debug']['enabled'] = True
        logger.info("启用调试模式")
    
    # 验证配置
    if not validate_config(config):
        logger.error("配置验证失败，请检查配置文件")
        return 1
    
    if args.dry_run:
        logger.info("配置验证完成，dry_run模式退出")
        return 0
    
    # 创建预处理器并运行
    try:
        preprocessor = ILDDataPreprocessor(config)
        logger.info("开始数据预处理...")
        preprocessor.run_preprocessing()
        logger.info("数据预处理完成！")
        
    except Exception as e:
        logger.error(f"数据预处理失败: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)