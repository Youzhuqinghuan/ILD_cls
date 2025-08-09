#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ILD分割推理程序
基于改进的MedSAM2模型进行ILD病变分割推理

主要改进：
1. 完全基于肺分割生成边界框，摆脱对Ground Truth的依赖
2. 支持背景切片（无病变）的检测和处理
3. 实现置信度计算机制，智能识别无病变状态
4. 支持单切片和批量患者推理
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from tqdm import tqdm
import logging
from skimage import measure
import SimpleITK as sitk

from sam2.build_sam import build_sam2
from sam2.utils.transforms import SAM2Transforms

# 设置随机种子
torch.manual_seed(2024)
torch.cuda.manual_seed(2024)
np.random.seed(2024)

# 病变类型映射
LESION_TYPE_MAPPING = {
    0: "background",
    1: "GGO",
    2: "reticulation", 
    3: "consolidation",
    4: "honeycombing"
}

class MedSAM2(nn.Module):
    """
    MedSAM2模型类，封装SAM2用于医学图像分割
    """
    def __init__(self, model):
        super().__init__()
        self.sam2_model = model
        # 冻结prompt encoder
        for param in self.sam2_model.sam_prompt_encoder.parameters():
            param.requires_grad = False
    
    def forward(self, image, box):
        """
        前向推理
        
        Args:
            image: 输入图像 (B, 3, 1024, 1024)
            box: 边界框 (B, 4) 或 (B, 2, 2)
        
        Returns:
            低分辨率分割logits
        """
        _features = self._image_encoder(image)
        img_embed, high_res_features = _features["image_embed"], _features["high_res_feats"]
        
        # 不计算prompt encoder的梯度
        with torch.no_grad():
            box_torch = torch.as_tensor(box, dtype=torch.float32, device=image.device)
            if len(box_torch.shape) == 2:
                box_coords = box_torch.reshape(-1, 2, 2)  # (B, 4) to (B, 2, 2)
                box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=image.device)
                box_labels = box_labels.repeat(box_torch.size(0), 1)
            concat_points = (box_coords, box_labels)

            sparse_embeddings, dense_embeddings = self.sam2_model.sam_prompt_encoder(
                points=concat_points,
                boxes=None,
                masks=None,
            )
        
        low_res_masks_logits, iou_predictions, sam_tokens_out, object_score_logits = self.sam2_model.sam_mask_decoder(
            image_embeddings=img_embed,
            image_pe=self.sam2_model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            repeat_image=False,
            high_res_features=high_res_features,
        )

        return low_res_masks_logits

    def _image_encoder(self, input_image):
        """
        图像编码器
        """
        backbone_out = self.sam2_model.forward_image(input_image)
        _, vision_feats, _, _ = self.sam2_model._prepare_backbone_features(backbone_out)
        if self.sam2_model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.sam2_model.no_mem_embed
        bb_feat_sizes = [(256, 256), (128, 128), (64, 64)]
        feats = [
            feat.permute(1, 2, 0).view(input_image.size(0), -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], bb_feat_sizes[::-1])
        ][::-1]

        _features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}
        return _features

class ILDInferenceEngine:
    """
    ILD推理引擎
    实现基于肺分割的边界框生成和置信度计算
    """
    
    def __init__(self, 
                 model_path: str,
                 config_path: str = "sam2_hiera_t.yaml",
                 sam2_checkpoint: str = "./checkpoints/sam2_hiera_tiny.pt",
                 lung_mask_dir: str = None,
                 device: str = "cuda:0",
                 confidence_threshold: float = 0.3,
                 bbox_padding: int = 5):
        """
        初始化推理引擎
        
        Args:
            model_path: 训练好的MedSAM2模型路径
            config_path: SAM2配置文件路径
            sam2_checkpoint: SAM2预训练模型路径
            lung_mask_dir: 肺分割文件目录（可选）
            device: 计算设备
            confidence_threshold: 置信度阈值
            bbox_padding: 边界框padding
        """
        self.device = torch.device(device)
        self.confidence_threshold = confidence_threshold
        self.bbox_padding = bbox_padding
        self.lung_mask_dir = lung_mask_dir
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        
        # 加载模型
        self._load_model(model_path, config_path, sam2_checkpoint)
        
        # SAM2变换
        self.sam2_transforms = SAM2Transforms(resolution=1024, mask_threshold=0)
        
        self.logger.info(f"推理引擎初始化完成，设备: {device}")
    
    def _load_model(self, model_path: str, config_path: str, sam2_checkpoint: str):
        """
        加载模型
        """
        try:
            # 构建SAM2模型
            sam2_model = build_sam2(config_path, sam2_checkpoint, device=self.device, mode="eval", apply_postprocessing=True)
            
            # 加载训练好的权重
            checkpoint = torch.load(model_path, map_location="cpu")
            
            # 创建MedSAM2模型
            self.model = MedSAM2(model=sam2_model)
            
            # 加载权重
            if "model" in checkpoint:
                self.model.load_state_dict(checkpoint["model"], strict=True)
            else:
                self.model.load_state_dict(checkpoint, strict=True)
            
            self.model.to(self.device)
            self.model.eval()
            
            self.logger.info(f"模型加载成功: {model_path}")
            
        except Exception as e:
            self.logger.error(f"模型加载失败: {e}")
            raise
    
    def _get_lung_bbox(self, lung_mask: np.ndarray, padding: int = None) -> np.ndarray:
        """
        基于肺分割生成边界框
        
        Args:
            lung_mask: 肺分割掩码 (H, W)
            padding: 边界框扩展像素数
            
        Returns:
            边界框坐标 [x_min, y_min, x_max, y_max]，已缩放到1024分辨率
        """
        if padding is None:
            padding = self.bbox_padding
        
        h, w = lung_mask.shape
        
        # 如果没有肺区域，返回整个图像的边界框
        if not np.any(lung_mask):
            scale_factor = 1024 / max(h, w)
            return np.array([0, 0, w, h]) * scale_factor
        
        # 找到肺区域的边界
        coords = np.where(lung_mask > 0)
        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()
        
        # 添加padding
        x_min = max(0, x_min - padding)
        x_max = min(w, x_max + padding)
        y_min = max(0, y_min - padding)
        y_max = min(h, y_max + padding)
        
        # 缩放到1024分辨率
        scale_factor = 1024 / max(h, w)
        bbox = np.array([x_min, y_min, x_max, y_max]) * scale_factor
        
        return bbox
    
    def _calculate_confidence(self, segmentation: np.ndarray, lung_mask: np.ndarray) -> float:
        """
        计算分割结果的置信度
        
        Args:
            segmentation: 分割结果 (H, W)
            lung_mask: 肺分割掩码 (H, W)
            
        Returns:
            置信度分数 [0, 1]
        """
        # 如果没有分割结果，置信度为0
        if not np.any(segmentation):
            return 0.0
        
        # 计算分割区域与肺区域的重叠度
        if np.any(lung_mask):
            overlap = np.sum((segmentation > 0) & (lung_mask > 0))
            seg_area = np.sum(segmentation > 0)
            overlap_ratio = overlap / seg_area if seg_area > 0 else 0.0
        else:
            overlap_ratio = 0.0
        
        # 连通组件分析
        labeled_seg = measure.label(segmentation)
        num_components = len(np.unique(labeled_seg)) - 1  # 减去背景
        
        # 计算最大连通组件的相对大小
        if num_components > 0:
            component_sizes = [np.sum(labeled_seg == i) for i in range(1, num_components + 1)]
            max_component_size = max(component_sizes)
            total_seg_area = np.sum(segmentation > 0)
            largest_component_ratio = max_component_size / total_seg_area if total_seg_area > 0 else 0.0
        else:
            largest_component_ratio = 0.0
        
        # 综合置信度计算
        confidence = 0.6 * overlap_ratio + 0.4 * largest_component_ratio
        
        # 如果连通组件过多，降低置信度
        if num_components > 5:
            confidence *= 0.8
        
        return min(1.0, max(0.0, confidence))
    
    @torch.no_grad()
    def _inference_with_bbox(self, features: Dict, bbox: np.ndarray, H: int, W: int) -> np.ndarray:
        """
        使用边界框进行推理
        
        Args:
            features: 图像特征
            bbox: 边界框 [x_min, y_min, x_max, y_max]
            H, W: 原始图像尺寸
            
        Returns:
            分割结果 (H, W)
        """
        img_embed, high_res_features = features["image_embed"], features["high_res_feats"]
        
        # 准备边界框
        box_torch = torch.as_tensor(bbox, dtype=torch.float32, device=self.device)
        if len(box_torch.shape) == 1:
            box_coords = box_torch.reshape(1, 2, 2)  # (4,) to (1, 2, 2)
        else:
            box_coords = box_torch.reshape(-1, 2, 2)
        
        box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=self.device)
        box_labels = box_labels.repeat(box_coords.size(0), 1)
        concat_points = (box_coords, box_labels)

        # Prompt编码
        sparse_embeddings, dense_embeddings = self.model.sam2_model.sam_prompt_encoder(
            points=concat_points,
            boxes=None,
            masks=None,
        )
        
        # 掩码解码
        low_res_masks_logits, iou_predictions, sam_tokens_out, object_score_logits = self.model.sam2_model.sam_mask_decoder(
            image_embeddings=img_embed,
            image_pe=self.model.sam2_model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            repeat_image=False,
            high_res_features=high_res_features,
        )

        # 后处理
        low_res_pred = torch.sigmoid(low_res_masks_logits)
        low_res_pred = F.interpolate(
            low_res_pred,
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )
        
        segmentation = (low_res_pred.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
        
        return segmentation
    
    def predict_single_slice(self, 
                           img_2d: np.ndarray, 
                           lung_mask_2d: np.ndarray = None, 
                           case_name: str = "unknown") -> Dict:
        """
        单切片推理
        
        Args:
            img_2d: 输入图像 (H, W) 或 (H, W, 3)
            lung_mask_2d: 肺分割掩码 (H, W)，可选
            case_name: 病例名称
            
        Returns:
            推理结果字典
        """
        try:
            # 图像预处理
            if len(img_2d.shape) == 2:
                img_3c = np.repeat(img_2d[:, :, None], 3, axis=-1)
            else:
                img_3c = img_2d
            
            H, W = img_3c.shape[:2]
            
            # 转换为SAM2输入格式
            img_1024_tensor = self.sam2_transforms(img_3c)[None, ...].to(self.device)
            
            # 获取图像特征
            features = self.model._image_encoder(img_1024_tensor)
            
            # 生成边界框
            if lung_mask_2d is not None and np.any(lung_mask_2d):
                bbox_1024 = self._get_lung_bbox(lung_mask_2d)
            else:
                # 如果没有肺分割，使用整个图像
                scale_factor = 1024 / max(H, W)
                bbox_1024 = np.array([0, 0, W, H]) * scale_factor
            
            # 推理
            segmentation = self._inference_with_bbox(features, bbox_1024, H, W)
            
            # 计算置信度
            if lung_mask_2d is not None:
                confidence = self._calculate_confidence(segmentation, lung_mask_2d)
            else:
                confidence = 0.5  # 默认置信度
            
            # 判断是否为背景切片
            is_background = confidence < self.confidence_threshold
            
            result = {
                'case_name': case_name,
                'segmentation': segmentation,
                'confidence': confidence,
                'is_background': is_background,
                'bbox': bbox_1024,
                'has_lesion': not is_background
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"单切片推理失败 {case_name}: {e}")
            return {
                'case_name': case_name,
                'segmentation': np.zeros((H, W), dtype=np.uint8),
                'confidence': 0.0,
                'is_background': True,
                'bbox': np.array([0, 0, W, H]),
                'has_lesion': False,
                'error': str(e)
            }
    
    def predict_patient(self, 
                       data_path: str, 
                       output_dir: str = None,
                       save_visualization: bool = False) -> Dict:
        """
        患者级推理
        
        Args:
            data_path: 患者数据路径（npz文件或数据目录）
            output_dir: 输出目录
            save_visualization: 是否保存可视化结果
            
        Returns:
            患者级推理结果
        """
        try:
            # 加载数据
            if data_path.endswith('.npz'):
                # 加载npz格式数据
                data = np.load(data_path, allow_pickle=True)
                img_3d = data['imgs']
                case_name = Path(data_path).stem
                
                # 尝试加载肺分割
                lung_masks_3d = None
                if self.lung_mask_dir:
                    lung_path = Path(self.lung_mask_dir) / f"{case_name}.npz"
                    if lung_path.exists():
                        lung_data = np.load(lung_path)
                        lung_masks_3d = lung_data.get('lung_masks', None)
            else:
                raise ValueError(f"不支持的数据格式: {data_path}")
            
            # 逐切片推理
            slice_results = []
            lesion_slices = []
            
            for z in range(img_3d.shape[0]):
                img_2d = img_3d[z]
                lung_mask_2d = lung_masks_3d[z] if lung_masks_3d is not None else None
                
                slice_result = self.predict_single_slice(
                    img_2d, lung_mask_2d, f"{case_name}_slice_{z:03d}"
                )
                
                slice_results.append(slice_result)
                
                if slice_result['has_lesion']:
                    lesion_slices.append(z)
            
            # 汇总患者级结果
            patient_result = {
                'case_name': case_name,
                'total_slices': len(slice_results),
                'lesion_slices': lesion_slices,
                'num_lesion_slices': len(lesion_slices),
                'has_lesion': len(lesion_slices) > 0,
                'slice_results': slice_results,
                'avg_confidence': np.mean([r['confidence'] for r in slice_results]),
                'max_confidence': np.max([r['confidence'] for r in slice_results])
            }
            
            # 保存结果
            if output_dir:
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                
                # 保存分割结果
                seg_3d = np.stack([r['segmentation'] for r in slice_results])
                np.savez_compressed(
                    output_path / f"{case_name}_segmentation.npz",
                    segmentation=seg_3d,
                    lesion_slices=lesion_slices
                )
                
                # 保存结果摘要
                summary = {
                    'case_name': case_name,
                    'has_lesion': patient_result['has_lesion'],
                    'num_lesion_slices': patient_result['num_lesion_slices'],
                    'lesion_slices': lesion_slices,
                    'avg_confidence': float(patient_result['avg_confidence']),
                    'max_confidence': float(patient_result['max_confidence'])
                }
                
                with open(output_path / f"{case_name}_summary.json", 'w') as f:
                    json.dump(summary, f, indent=2)
            
            return patient_result
            
        except Exception as e:
            self.logger.error(f"患者推理失败 {data_path}: {e}")
            return {
                'case_name': Path(data_path).stem,
                'error': str(e),
                'has_lesion': False
            }

def main():
    """
    主函数
    """
    import yaml
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='ILD病变分割推理程序')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='配置文件路径 (默认: config.yaml)')
    args = parser.parse_args()
    
    # 读取配置文件
    config_path = args.config
    if not Path(config_path).exists():
        print(f"错误: 配置文件 {config_path} 不存在")
        return
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 检查配置文件结构
    if 'inference' not in config:
        print("错误: 配置文件中缺少 'inference' 部分")
        return
    
    inference_config = config['inference']
    
    # 从配置文件中提取参数
    model_config = inference_config.get('model', {})
    data_config = inference_config.get('data', {})
    params_config = inference_config.get('params', {})
    output_config = inference_config.get('output', {})
    
    # 模型参数
    model_path = model_config.get('model_path')
    config_path = model_config.get('config_path', 'sam2_hiera_t.yaml')
    sam2_checkpoint = model_config.get('sam2_checkpoint', './checkpoints/sam2_hiera_tiny.pt')
    
    # 数据参数
    data_root = data_config.get('data_root')
    lung_mask_dir = data_config.get('lung_mask_dir')
    output_dir = data_config.get('output_dir')
    
    # 推理参数
    device = params_config.get('device', 'cuda:0')
    confidence_threshold = params_config.get('confidence_threshold', 0.3)
    bbox_padding = params_config.get('bbox_padding', 5)
    batch_size = params_config.get('batch_size', 1)
    
    # 输出参数
    save_visualization = output_config.get('save_visualization', False)
    verbose = output_config.get('verbose', False)
    
    # 检查必需参数
    if not model_path:
        print("错误: 配置文件中缺少模型路径 (inference.model.model_path)")
        return
    if not data_root:
        print("错误: 配置文件中缺少数据根目录 (inference.data.data_root)")
        return
    if not output_dir:
        print("错误: 配置文件中缺少输出目录 (inference.data.output_dir)")
        return
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    try:
        # 创建推理引擎
        engine = ILDInferenceEngine(
            model_path=model_path,
            config_path=config_path,
            sam2_checkpoint=sam2_checkpoint,
            lung_mask_dir=lung_mask_dir,
            device=device,
            confidence_threshold=confidence_threshold,
            bbox_padding=bbox_padding
        )
        
        # 创建输出目录
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 获取数据文件列表
        data_path = Path(data_root)
        if data_path.is_file() and data_path.suffix == '.npz':
            # 单个文件
            data_files = [data_path]
        else:
            # 目录中的所有npz文件
            data_files = list(data_path.glob('*.npz'))
        
        if not data_files:
            logger.error(f"在 {data_path} 中未找到npz文件")
            return
        
        logger.info(f"找到 {len(data_files)} 个数据文件")
        
        # 批量推理
        all_results = []
        lesion_count = 0
        no_lesion_count = 0
        
        for data_file in tqdm(data_files, desc="处理患者数据"):
            result = engine.predict_patient(
                str(data_file),
                str(output_path),
                save_visualization
            )
            
            all_results.append(result)
            
            if result.get('has_lesion', False):
                lesion_count += 1
            else:
                no_lesion_count += 1
        
        # 保存汇总结果
        summary_results = {
            'total_cases': len(data_files),
            'lesion_detected': lesion_count,
            'no_lesion': no_lesion_count,
            'detection_rate': lesion_count / len(data_files) if data_files else 0,
            'case_results': [
                {
                    'case_name': r['case_name'],
                    'has_lesion': r.get('has_lesion', False),
                    'num_lesion_slices': r.get('num_lesion_slices', 0),
                    'avg_confidence': r.get('avg_confidence', 0.0),
                    'max_confidence': r.get('max_confidence', 0.0)
                }
                for r in all_results
            ]
        }
        
        with open(output_dir / 'inference_summary.json', 'w') as f:
            json.dump(summary_results, f, indent=2)
        
        logger.info(f"推理完成！")
        logger.info(f"总病例数: {len(data_files)}")
        logger.info(f"检测到病变: {lesion_count}")
        logger.info(f"无病变: {no_lesion_count}")
        logger.info(f"检测率: {lesion_count/len(data_files)*100:.1f}%")
        logger.info(f"结果保存至: {output_dir}")
        
    except Exception as e:
        logger.error(f"推理过程出错: {e}")
        raise

if __name__ == "__main__":
    main()