#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ILD分割模型训练程序
基于SAM2架构，支持肺分割驱动的边界框生成和背景切片处理

主要改进：
1. 使用新的ILDDataset类，支持肺分割驱动的边界框生成
2. 支持背景切片训练，提升模型鲁棒性
3. 实现加权损失函数，平衡背景和病变类别
4. 增强训练监控，分别记录背景和病变切片性能
5. 通过配置文件管理所有参数
"""

import os
import sys
import yaml
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import monai
import cv2
import argparse
import random
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import shutil

# SwanLab监控
try:
    import swanlab
    SWANLAB_AVAILABLE = True
except ImportError:
    SWANLAB_AVAILABLE = False
    print("SwanLab未安装，将跳过实验监控")

# 导入SAM2相关模块
from sam2.build_sam import build_sam2
from sam2.utils.transforms import SAM2Transforms

# 导入自定义数据集
from dataset import ILDDataset, create_dataloader


def set_seed(seed: int):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def setup_environment():
    """设置环境变量"""
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["OPENBLAS_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "6"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
    os.environ["NUMEXPR_NUM_THREADS"] = "6"


def setup_logging(log_dir: Path, log_level: str = "INFO"):
    """设置日志系统"""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建日志文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"train_{timestamp}.log"
    
    # 配置日志
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


def load_config(config_path: str) -> Dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def show_mask(mask, ax, random_color=False):
    """显示分割掩码"""
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251 / 255, 252 / 255, 30 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax):
    """显示边界框"""
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="blue", facecolor=(0, 0, 0, 0), lw=2)
    )


class MedSAM2(nn.Module):
    """MedSAM2模型类，封装SAM2用于医学图像分割"""
    
    def __init__(self, model):
        super().__init__()
        self.sam2_model = model
        # 冻结prompt encoder
        for param in self.sam2_model.sam_prompt_encoder.parameters():
            param.requires_grad = False
    
    def forward(self, image, box):
        """
        前向传播
        
        Args:
            image: (B, 3, 1024, 1024)
            box: (B, 4) - [x_min, y_min, x_max, y_max]
        
        Returns:
            low_res_masks_logits: 低分辨率掩码logits
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
            image_embeddings=img_embed,  # (B, 256, 64, 64)
            image_pe=self.sam2_model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            repeat_image=False,
            high_res_features=high_res_features,
        )
        
        return low_res_masks_logits
    
    def _image_encoder(self, input_image):
        """图像编码器"""
        backbone_out = self.sam2_model.forward_image(input_image)
        _, vision_feats, _, _ = self.sam2_model._prepare_backbone_features(backbone_out)
        
        # 添加no_mem_embed
        if self.sam2_model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.sam2_model.no_mem_embed
        
        bb_feat_sizes = [(256, 256), (128, 128), (64, 64)]
        feats = [
            feat.permute(1, 2, 0).view(input_image.size(0), -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], bb_feat_sizes[::-1])
        ][::-1]
        
        _features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}
        return _features


class WeightedLoss(nn.Module):
    """加权损失函数，平衡背景和病变类别"""
    
    def __init__(self, background_weight: float = 0.5, lesion_weight: float = 1.0, 
                 dice_weight: float = 1.0, ce_weight: float = 1.0):
        super().__init__()
        self.background_weight = background_weight
        self.lesion_weight = lesion_weight
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        
        # 创建损失函数
        self.dice_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
        
        # BCE损失，使用pos_weight平衡正负样本
        pos_weight = torch.tensor([lesion_weight / background_weight])
        self.ce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="mean")
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor, is_background: torch.Tensor = None):
        """
        计算加权损失
        
        Args:
            pred: 预测结果 (B, 1, H, W)
            target: 真实标签 (B, 1, H, W)
            is_background: 背景标志 (B, 1)，可选
        
        Returns:
            总损失
        """
        dice_loss = self.dice_loss(pred, target)
        ce_loss = self.ce_loss(pred, target.float())
        
        total_loss = self.dice_weight * dice_loss + self.ce_weight * ce_loss
        
        return total_loss


def calculate_metrics(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    """计算评估指标"""
    pred_binary = (torch.sigmoid(pred) > threshold).float()
    target_binary = target.float()
    
    # 计算基本指标
    tp = (pred_binary * target_binary).sum().item()
    fp = (pred_binary * (1 - target_binary)).sum().item()
    fn = ((1 - pred_binary) * target_binary).sum().item()
    tn = ((1 - pred_binary) * (1 - target_binary)).sum().item()
    
    # 避免除零
    eps = 1e-8
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    specificity = tn / (tn + fp + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    
    # Dice系数
    dice = 2 * tp / (2 * tp + fp + fn + eps)
    
    return {
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1': f1,
        'dice': dice
    }


def validate(model: MedSAM2, val_dataloader, criterion: WeightedLoss, device: torch.device, logger) -> Dict[str, float]:
    """验证函数"""
    model.eval()
    val_loss = 0
    all_metrics = {'lesion': [], 'background': [], 'overall': []}
    
    with torch.no_grad():
        for step, (image, gt2D, boxes, filenames, is_background) in enumerate(tqdm(val_dataloader, desc="Validating")):
            boxes_np = boxes.detach().cpu().numpy()
            image, gt2D, is_background = image.to(device), gt2D.to(device), is_background.to(device)
            
            # 前向传播
            medsam_pred = model(image, boxes_np)
            loss = criterion(medsam_pred, gt2D, is_background)
            val_loss += loss.item()
            
            # 计算指标
            batch_metrics = calculate_metrics(medsam_pred, gt2D)
            all_metrics['overall'].append(batch_metrics)
            
            # 分别计算背景和病变切片的指标
            background_mask = is_background.squeeze() > 0.5
            lesion_mask = ~background_mask
            
            if torch.any(background_mask):
                bg_metrics = calculate_metrics(
                    medsam_pred[background_mask], 
                    gt2D[background_mask]
                )
                all_metrics['background'].append(bg_metrics)
            
            if torch.any(lesion_mask):
                lesion_metrics = calculate_metrics(
                    medsam_pred[lesion_mask], 
                    gt2D[lesion_mask]
                )
                all_metrics['lesion'].append(lesion_metrics)
    
    # 计算平均指标
    avg_metrics = {}
    for category, metrics_list in all_metrics.items():
        if metrics_list:
            avg_metrics[category] = {
                metric: np.mean([m[metric] for m in metrics_list])
                for metric in metrics_list[0].keys()
            }
    
    val_loss /= (step + 1)
    
    # 记录验证结果
    logger.info(f"Validation Loss: {val_loss:.4f}")
    for category, metrics in avg_metrics.items():
        logger.info(f"{category.capitalize()} metrics: {metrics}")
    
    model.train()
    return {'loss': val_loss, 'metrics': avg_metrics}


def perform_sanity_check(dataset, save_dir: Path, logger):
    """数据安全检查"""
    logger.info("执行数据安全检查...")
    
    # 创建数据加载器
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
    images, gts, bboxes, names, is_background = next(iter(dataloader))
    
    # 创建逆变换
    inv_sam2_transform = torchvision.transforms.Compose([
        torchvision.transforms.Normalize(mean=[0, 0, 0], std=[1 / i for i in dataset._transform.std]),
        torchvision.transforms.Normalize(mean=[-1 * i for i in dataset._transform.mean], std=[1, 1, 1]),
    ])
    
    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(20, 20))
    axes = axes.flatten()
    
    for i in range(min(4, len(images))):
        ax = axes[i]
        
        # 显示图像
        img_display = inv_sam2_transform(images[i].clone()).permute(1, 2, 0).numpy()
        ax.imshow(img_display)
        
        # 显示掩码
        mask_resized = cv2.resize(
            gts[i].squeeze(0).numpy(),
            (1024, 1024),
            interpolation=cv2.INTER_NEAREST
        )
        show_mask(mask_resized, ax)
        
        # 显示边界框
        show_box(bboxes[i].numpy(), ax)
        
        # 设置标题
        bg_flag = "[BG]" if is_background[i] > 0.5 else "[LESION]"
        ax.set_title(f"{names[i]} {bg_flag}")
        ax.axis("off")
    
    plt.tight_layout()
    plt.savefig(save_dir / "data_sanitycheck.png", bbox_inches="tight", dpi=300)
    plt.close()
    
    logger.info(f"数据安全检查完成，结果保存至: {save_dir / 'data_sanitycheck.png'}")


def main():
    """主训练函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="ILD分割模型训练")
    parser.add_argument("-c", "--config", type=str, default="config.yaml", help="配置文件路径")
    parser.add_argument("--resume", type=str, default=None, help="恢复训练的检查点路径")
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置环境
    setup_environment()
    set_seed(config['global']['seed'])
    
    # 创建工作目录（使用北京时间）
    beijing_tz = timezone(timedelta(hours=8))
    timestamp = datetime.now(beijing_tz).strftime("%Y%m%d_%H%M%S")
    
    # 工作目录设置在ILD_Segmentation_New下
    current_dir = Path(__file__).parent
    work_dir = current_dir / "workspace"
    
    # 根据训练参数创建子目录
    batch_size = config['training']['params']['batch_size']
    learning_rate = config['training']['params']['learning_rate']
    epochs = config['training']['params']['epochs']
    
    exp_name = f"bs{batch_size}_lr{learning_rate}_ep{epochs}_{timestamp}"
    model_save_path = work_dir / exp_name
    model_save_path.mkdir(parents=True, exist_ok=True)
    
    # 设置日志
    logger = setup_logging(model_save_path / "logs", config['global']['log_level'])
    logger.info(f"开始训练，工作目录: {model_save_path}")
    
    # 初始化SwanLab监控
    if SWANLAB_AVAILABLE:
        swanlab.init(
            project="ILD_Segmentation",
            experiment_name=exp_name,
            description="ILD分割模型训练 - SAM2架构",
            config={
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "epochs": epochs,
                "model": config['training']['model']['name'],
                "bbox_shift": config['training']['dataloader'].get('bbox_shift', 20),
                "dice_weight": config['training']['loss']['dice_weight'],
                "ce_weight": config['training']['loss']['ce_weight'],
                "background_weight": config['training']['loss'].get('background_weight', 0.5),
                "lesion_weight": config['training']['loss'].get('lesion_weight', 1.0)
            }
        )
        logger.info("SwanLab监控已启动")
    else:
        logger.info("SwanLab未可用，跳过实验监控")
    
    # 保存配置文件副本
    shutil.copy(args.config, model_save_path / "config.yaml")
    shutil.copy(__file__, model_save_path / "train.py")
    
    # 设置设备
    device = torch.device(config['global']['device'])
    logger.info(f"使用设备: {device}")
    
    # 创建数据集和数据加载器
    logger.info("创建数据集...")
    
    # 训练数据集
    train_dataset = ILDDataset(
        data_root=config['data']['preprocessed']['output_dir'],
        split='train',
        bbox_shift=config['training']['dataloader'].get('bbox_shift', 20),
        background_sample_ratio=config['preprocessing']['background_slices']['ratios']['train'],
        enable_background_training=True
    )
    
    train_dataloader = create_dataloader(
        data_root=config['data']['preprocessed']['output_dir'],
        split='train',
        batch_size=config['training']['params']['batch_size'],
        num_workers=config['training']['dataloader']['num_workers'],
        bbox_shift=config['training']['dataloader'].get('bbox_shift', 20),
        background_sample_ratio=config['preprocessing']['background_slices']['ratios']['train'],
        enable_background_training=True,
        lesion_type_mapping=config['data']['lesion_type_mapping']
    )
    
    # 验证数据集
    val_dataset = ILDDataset(
        data_root=config['data']['preprocessed']['output_dir'],
        split='val',
        bbox_shift=0,  # 验证时不使用随机扰动
        enable_background_training=True
    )
    
    val_dataloader = create_dataloader(
        data_root=config['data']['preprocessed']['output_dir'],
        split='val',
        batch_size=config['training']['params']['batch_size'],
        num_workers=config['training']['dataloader']['num_workers'],
        bbox_shift=0,
        enable_background_training=True,
        lesion_type_mapping=config['data']['lesion_type_mapping']
    )
    
    logger.info(f"训练集样本数: {len(train_dataset)}")
    logger.info(f"验证集样本数: {len(val_dataset)}")
    logger.info(f"训练集统计: {train_dataset.get_statistics()}")
    logger.info(f"验证集统计: {val_dataset.get_statistics()}")
    
    # 执行数据安全检查
    perform_sanity_check(train_dataset, model_save_path, logger)
    
    # 构建模型
    logger.info("构建模型...")
    model_cfg = config['training']['model']['config_file']
    sam2_checkpoint = config['training']['model']['checkpoint']
    
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=True)
    medsam_model = MedSAM2(model=sam2_model)
    medsam_model.to(device)
    medsam_model.train()
    
    # 打印模型参数信息
    total_params = sum(p.numel() for p in medsam_model.parameters())
    trainable_params = sum(p.numel() for p in medsam_model.parameters() if p.requires_grad)
    logger.info(f"总参数数量: {total_params:,}")
    logger.info(f"可训练参数数量: {trainable_params:,}")
    
    # 设置优化器（仅训练image encoder和mask decoder）
    img_mask_encdec_params = list(medsam_model.sam2_model.image_encoder.parameters()) + \
                           list(medsam_model.sam2_model.sam_mask_decoder.parameters())
    
    optimizer = torch.optim.AdamW(
        img_mask_encdec_params,
        lr=float(config['training']['params']['learning_rate']),
        weight_decay=float(config['training']['params']['weight_decay'])
    )
    
    trainable_encdec_params = sum(p.numel() for p in img_mask_encdec_params if p.requires_grad)
    logger.info(f"编码器和解码器可训练参数数量: {trainable_encdec_params:,}")
    
    # 创建损失函数
    criterion = WeightedLoss(
        background_weight=config['training']['loss'].get('background_weight', 0.5),
        lesion_weight=config['training']['loss'].get('lesion_weight', 1.0),
        dice_weight=config['training']['loss']['dice_weight'],
        ce_weight=config['training']['loss']['ce_weight']
    )
    criterion.to(device)
    
    # 训练参数
    num_epochs = config['training']['params']['epochs']
    save_interval = config['training']['checkpoint'].get('save_interval', 50)
    
    # 学习率调度器：线性Warmup + 余弦退火
    scheduler_config = config['training'].get('scheduler', {})
    warmup_ratio = scheduler_config.get('warmup_ratio', 0.1)
    min_lr_ratio = scheduler_config.get('min_lr_ratio', 0.01)
    
    total_steps = len(train_dataloader) * num_epochs
    warmup_steps = int(total_steps * warmup_ratio)
    base_lr = float(config['training']['params']['learning_rate'])
    min_lr = base_lr * min_lr_ratio
    
    # 创建线性Warmup调度器
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.01,  # 从1%的学习率开始
        end_factor=1.0,     # 到达100%学习率
        total_iters=warmup_steps
    )
    
    # 创建余弦退火调度器
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps,
        eta_min=min_lr
    )
    
    # 组合调度器
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps]
    )
    
    logger.info(f"学习率调度器设置:")
    logger.info(f"  类型: {scheduler_config.get('type', 'cosine_annealing')}")
    logger.info(f"  总步数: {total_steps}")
    logger.info(f"  Warmup步数: {warmup_steps} ({warmup_ratio*100:.1f}%)")
    logger.info(f"  初始学习率: {base_lr:.2e}")
    logger.info(f"  最小学习率: {min_lr:.2e}")
    
    # 训练历史记录
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    # 恢复训练
    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        logger.info(f"从检查点恢复训练: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        start_epoch = checkpoint["epoch"] + 1
        medsam_model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        
        # 恢复调度器状态
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
            logger.info("已恢复学习率调度器状态")
        
        best_val_loss = checkpoint.get("best_val_loss", float('inf'))
        train_losses = checkpoint.get("train_losses", [])
        val_losses = checkpoint.get("val_losses", [])
        logger.info(f"从第{start_epoch}轮开始恢复训练")
    
    # 开始训练
    logger.info("开始训练循环...")
    
    for epoch in range(start_epoch, num_epochs):
        # 训练阶段
        medsam_model.train()
        epoch_loss = 0
        epoch_metrics = {'lesion': [], 'background': [], 'overall': []}
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for step, (image, gt2D, boxes, filenames, is_background) in enumerate(progress_bar):
            optimizer.zero_grad()
            
            # 数据移动到设备
            boxes_np = boxes.detach().cpu().numpy()
            image, gt2D, is_background = image.to(device), gt2D.to(device), is_background.to(device)
            
            # 前向传播
            medsam_pred = medsam_model(image, boxes_np)
            loss = criterion(medsam_pred, gt2D, is_background)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # 计算训练指标
            with torch.no_grad():
                batch_metrics = calculate_metrics(medsam_pred, gt2D)
                epoch_metrics['overall'].append(batch_metrics)
                
                # 分别计算背景和病变切片的指标
                background_mask = is_background.squeeze() > 0.5
                lesion_mask = ~background_mask
                
                step_metrics = {}
                if torch.any(background_mask):
                    bg_metrics = calculate_metrics(
                        medsam_pred[background_mask], 
                        gt2D[background_mask]
                    )
                    epoch_metrics['background'].append(bg_metrics)
                    step_metrics.update({f"train/background_{k}": v for k, v in bg_metrics.items()})
                
                if torch.any(lesion_mask):
                    lesion_metrics = calculate_metrics(
                        medsam_pred[lesion_mask], 
                        gt2D[lesion_mask]
                    )
                    epoch_metrics['lesion'].append(lesion_metrics)
                    step_metrics.update({f"train/lesion_{k}": v for k, v in lesion_metrics.items()})
            
            # 学习率调度
            global_step = step + epoch * len(train_dataloader)
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            
            # SwanLab按步记录训练指标
            if SWANLAB_AVAILABLE:
                log_data = {
                    'train/loss': loss.item(),
                    'train/learning_rate': current_lr,
                    'epoch': epoch + 1,
                    'global_step': global_step,
                    'warmup_progress': min(global_step / warmup_steps, 1.0) if warmup_steps > 0 else 1.0
                }
                log_data.update(step_metrics)
                swanlab.log(log_data, step=global_step)
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{current_lr:.2e}',
                'step': global_step
            })
        
        # 计算平均损失和指标
        avg_train_loss = epoch_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)
        
        # 计算平均训练指标
        avg_train_metrics = {}
        for category, metrics_list in epoch_metrics.items():
            if metrics_list:
                avg_train_metrics[category] = {
                    metric: np.mean([m[metric] for m in metrics_list])
                    for metric in metrics_list[0].keys()
                }
        
        # 验证阶段
        val_results = validate(medsam_model, val_dataloader, criterion, device, logger)
        val_loss = val_results['loss']
        val_losses.append(val_loss)
        
        # 记录训练结果
        logger.info(f"Epoch {epoch+1}/{num_epochs}:")
        logger.info(f"  训练损失: {avg_train_loss:.4f}")
        logger.info(f"  验证损失: {val_loss:.4f}")
        
        for category, metrics in avg_train_metrics.items():
            logger.info(f"  训练{category}指标: {metrics}")
        
        # SwanLab监控记录验证指标
        if SWANLAB_AVAILABLE:
            # 记录验证损失和指标
            val_log_data = {
                "val/loss": val_loss,
                "epoch": epoch + 1
            }
            
            # 记录验证指标
            val_metrics = val_results.get('metrics', {})
            for category, metrics in val_metrics.items():
                for metric_name, metric_value in metrics.items():
                    val_log_data[f"val/{category}_{metric_name}"] = metric_value
            
            swanlab.log(val_log_data)
        
        # 保存最新模型
        checkpoint = {
            "model": medsam_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "best_val_loss": best_val_loss,
            "config": config
        }
        torch.save(checkpoint, model_save_path / "latest_checkpoint.pth")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint["best_val_loss"] = best_val_loss
            torch.save(checkpoint, model_save_path / "best_checkpoint.pth")
            logger.info(f"  保存最佳模型，验证损失: {best_val_loss:.4f}")
        
        # 定期保存检查点
        if (epoch + 1) % save_interval == 0:
            torch.save(checkpoint, model_save_path / f"checkpoint_epoch_{epoch+1}.pth")
        
        # 绘制损失曲线
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Training Loss', color='blue')
        plt.plot(val_losses, label='Validation Loss', color='red')
        plt.title("Training and Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        if len(train_losses) > 10:
            plt.plot(train_losses[-10:], label='Training Loss (Last 10)', color='blue')
            plt.plot(val_losses[-10:], label='Validation Loss (Last 10)', color='red')
            plt.title("Recent Loss Trend")
            plt.xlabel("Recent Epochs")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(model_save_path / "training_curves.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    logger.info("训练完成！")
    logger.info(f"最佳验证损失: {best_val_loss:.4f}")
    logger.info(f"模型保存路径: {model_save_path}")
    
    # 关闭SwanLab监控
    if SWANLAB_AVAILABLE:
        swanlab.finish()
        logger.info("SwanLab监控已关闭")


if __name__ == "__main__":
    main()