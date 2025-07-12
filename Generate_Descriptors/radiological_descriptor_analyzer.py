#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
放射学描述符分析程序
基于规则的算法分析病变的轴向分布和整体征象

轴向分布：外周、中心、弥漫/散在
整体征象：胸膜下、胸膜下省略
"""

import os
import json
import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi
from skimage.morphology import binary_closing, disk
from skimage.measure import label, regionprops
from skimage.segmentation import clear_border
import argparse
from tqdm import tqdm
import glob


def get_lung_mask_from_ct(ct_array, spacing, threshold=-300):
    """
    从CT图像生成整肺掩码
    
    Args:
        ct_array: CT图像数组 (Z, Y, X)
        spacing: 像素间距
        threshold: 阈值
    
    Returns:
        lung_mask: 整肺二值掩码
    """
    lung_mask = np.zeros_like(ct_array, dtype=bool)
    
    # 对每个切片进行肺部分割
    for i in range(ct_array.shape[0]):
        slice_img = ct_array[i]
        
        # 二值化
        binary = slice_img < threshold
        
        # 清除边界
        cleared = clear_border(binary)
        
        # 标记连通区域
        label_image = label(cleared)
        
        # 保留两个最大的连通区域（左右肺）
        areas = [r.area for r in regionprops(label_image)]
        if len(areas) >= 2:
            areas.sort()
            for region in regionprops(label_image):
                if region.area < areas[-2]:
                    for coordinates in region.coords:
                        label_image[coordinates[0], coordinates[1]] = 0
        
        binary = label_image > 0
        
        # 形态学闭运算
        if np.any(binary):
            selem = disk(3)
            binary = binary_closing(binary, selem)
        
        lung_mask[i] = binary
    
    return lung_mask


def analyze_axial_distribution(disease_mask, lung_mask, spacing):
    """
    分析病变的轴向分布
    
    Args:
        disease_mask: 病变掩码
        lung_mask: 整肺掩码
        spacing: 像素间距 (z, y, x)
    
    Returns:
        axial_distribution: 轴向分布类别
        metrics: 相关指标
    """
    # 计算距离变换
    dist_transform = ndi.distance_transform_edt(lung_mask, sampling=spacing)
    
    # 获取病变区域的距离值
    disease_distances = dist_transform[disease_mask > 0]
    
    if len(disease_distances) == 0:
        return "unknown", {}
    
    # 计算最大半径（用于归一化）
    max_radius = dist_transform.max()
    
    # 规则参数
    per_mm = 10  # 外周阈值（毫米）
    per_rel = 0.20  # 外周相对阈值
    
    # 计算外周体素
    peripheral_voxels = (disease_distances <= per_mm) | (disease_distances / max_radius <= per_rel)
    central_voxels = ~peripheral_voxels
    
    # 计算比例
    peripheral_ratio = peripheral_voxels.mean()
    central_ratio = central_voxels.mean()
    
    # 分类规则
    if peripheral_ratio >= 0.70:
        axial_distribution = "peripheral"
    elif central_ratio >= 0.70:
        axial_distribution = "central"
    else:
        axial_distribution = "diffuse_scattered"
    
    metrics = {
        "peripheral_ratio": float(peripheral_ratio),
        "central_ratio": float(central_ratio),
        "mean_distance_mm": float(disease_distances.mean()),
        "min_distance_mm": float(disease_distances.min()),
        "max_distance_mm": float(disease_distances.max()),
        "max_radius_mm": float(max_radius)
    }
    
    return axial_distribution, metrics


def analyze_manifestation(disease_mask, lung_mask, spacing):
    """
    分析整体征象（胸膜下受累）
    
    Args:
        disease_mask: 病变掩码
        lung_mask: 整肺掩码
        spacing: 像素间距
    
    Returns:
        manifestation: 整体征象类别
        metrics: 相关指标
    """
    # 计算距离变换
    dist_transform = ndi.distance_transform_edt(lung_mask, sampling=spacing)
    
    # 获取病变区域的距离值
    disease_distances = dist_transform[disease_mask > 0]
    
    if len(disease_distances) == 0:
        return "unknown", {}
    
    # 胸膜下阈值（3mm）
    subpleural_threshold = 3.0
    
    # 判断是否有胸膜下受累
    min_distance = disease_distances.min()
    subpleural_involved = min_distance <= subpleural_threshold
    
    # 计算胸膜下体素比例
    subpleural_voxels = disease_distances <= subpleural_threshold
    subpleural_ratio = subpleural_voxels.mean()
    
    # 分类规则
    if subpleural_involved:
        manifestation = "subpleural"
    else:
        manifestation = "subpleural_omitted"
    
    metrics = {
        "min_distance_to_pleura_mm": float(min_distance),
        "subpleural_ratio": float(subpleural_ratio),
        "subpleural_involved": bool(subpleural_involved)
    }
    
    return manifestation, metrics


def process_case(ct_path, disease_mask_path, case_name):
    """
    处理单个病例
    
    Args:
        ct_path: CT图像路径
        disease_mask_path: 病变掩码路径
        case_name: 病例名称
    
    Returns:
        result: 分析结果字典
    """
    result = {
        "case_name": case_name,
        "ct_path": ct_path,
        "disease_mask_path": disease_mask_path,
        "success": False,
        "error": None
    }
    
    try:
        # 读取CT图像
        ct_sitk = sitk.ReadImage(ct_path)
        ct_array = sitk.GetArrayFromImage(ct_sitk)
        spacing = ct_sitk.GetSpacing()[::-1]  # 转换为 (z, y, x) 顺序
        
        # 读取病变掩码
        disease_sitk = sitk.ReadImage(disease_mask_path)
        disease_mask = sitk.GetArrayFromImage(disease_sitk) > 0
        
        print(f"Processing {case_name}...")
        print(f"CT shape: {ct_array.shape}, Disease mask shape: {disease_mask.shape}")
        print(f"Spacing: {spacing}")
        
        # 检查病变掩码是否为空
        if not np.any(disease_mask):
            result["error"] = "Empty disease mask"
            return result
        
        # 生成整肺掩码
        lung_mask = get_lung_mask_from_ct(ct_array, spacing)
        
        if not np.any(lung_mask):
            result["error"] = "Failed to generate lung mask"
            return result
        
        # 确保病变掩码在肺部范围内
        disease_mask = disease_mask & lung_mask
        
        if not np.any(disease_mask):
            result["error"] = "No disease voxels within lung mask"
            return result
        
        # 分析轴向分布
        axial_distribution, axial_metrics = analyze_axial_distribution(
            disease_mask, lung_mask, spacing
        )
        
        # 分析整体征象
        manifestation, manifestation_metrics = analyze_manifestation(
            disease_mask, lung_mask, spacing
        )
        
        # 更新结果
        result.update({
            "success": True,
            "axial_distribution": axial_distribution,
            "manifestation": manifestation,
            "axial_metrics": axial_metrics,
            "manifestation_metrics": manifestation_metrics,
            "disease_voxel_count": int(np.sum(disease_mask)),
            "lung_voxel_count": int(np.sum(lung_mask))
        })
        
        print(f"Results: Axial={axial_distribution}, Manifestation={manifestation}")
        
    except Exception as e:
        result["error"] = str(e)
        print(f"Error processing {case_name}: {e}")
    
    return result


def find_matching_files(images_dir, labels_dir):
    """
    查找匹配的CT图像和病变掩码文件
    
    Args:
        images_dir: CT图像目录
        labels_dir: 病变掩码目录
    
    Returns:
        matches: 匹配的文件对列表
    """
    matches = []
    
    # 获取所有病变掩码文件
    label_files = glob.glob(os.path.join(labels_dir, "CT_*.nii.gz"))
    
    for label_path in label_files:
        label_filename = os.path.basename(label_path)
        # 从 CT_NAME.nii.gz 提取 NAME
        case_name = label_filename.replace("CT_", "").replace(".nii.gz", "")
        
        # 查找对应的CT图像 CT_NAME_0000.nii.gz
        ct_filename = f"CT_{case_name}_0000.nii.gz"
        ct_path = os.path.join(images_dir, ct_filename)
        
        if os.path.exists(ct_path):
            matches.append({
                "case_name": case_name,
                "ct_path": ct_path,
                "label_path": label_path
            })
            print(f"Found match: {case_name}")
        else:
            print(f"Warning: No matching CT image for {label_filename}")
    
    return matches


def save_results(results, output_path):
    """
    保存结果为JSONL格式
    
    Args:
        results: 结果列表
        output_path: 输出文件路径
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for result in results:
            if result["success"]:
                # 只保存成功的结果和关键信息
                output_record = {
                    "case_name": result["case_name"],
                    "axial_distribution": result["axial_distribution"],
                    "manifestation": result["manifestation"],
                    "axial_metrics": result["axial_metrics"],
                    "manifestation_metrics": result["manifestation_metrics"]
                }
            else:
                # 保存失败的记录
                output_record = {
                    "case_name": result["case_name"],
                    "error": result["error"]
                }
            
            f.write(json.dumps(output_record, ensure_ascii=False) + '\n')
    
    print(f"Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='放射学描述符分析程序')
    parser.add_argument('--images_dir', type=str,
                       default='/home/huchengpeng/workspace/dataset/images',
                       help='CT图像目录')
    parser.add_argument('--labels_dir', type=str,
                       default='/home/huchengpeng/workspace/dataset/labels',
                       help='病变掩码目录')
    parser.add_argument('--output_dir', type=str,
                       default='/home/huchengpeng/workspace/Generate_Descriptors',
                       help='输出目录')
    parser.add_argument('--test_single', type=str, default=None,
                       help='测试单个病例，例如：NSIP1')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.test_single:
        # 测试单个病例
        case_name = args.test_single
        ct_path = os.path.join(args.images_dir, f"CT_{case_name}_0000.nii.gz")
        label_path = os.path.join(args.labels_dir, f"CT_{case_name}.nii.gz")
        
        if not os.path.exists(ct_path):
            print(f"Error: CT image not found: {ct_path}")
            return
        
        if not os.path.exists(label_path):
            print(f"Error: Label not found: {label_path}")
            return
        
        print(f"Testing single case: {case_name}")
        result = process_case(ct_path, label_path, case_name)
        
        # 保存单个结果
        output_path = os.path.join(args.output_dir, f"radiological_descriptors_{case_name}.jsonl")
        save_results([result], output_path)
        
    else:
        # 批量处理
        print("Finding matching files...")
        matches = find_matching_files(args.images_dir, args.labels_dir)
        
        if not matches:
            print("No matching files found!")
            return
        
        print(f"Found {len(matches)} matching cases")
        
        # 处理所有病例
        results = []
        for match in tqdm(matches, desc="Processing cases"):
            result = process_case(
                match["ct_path"],
                match["label_path"],
                match["case_name"]
            )
            results.append(result)
        
        # 保存所有结果
        output_path = os.path.join(args.output_dir, "radiological_descriptors_all.jsonl")
        save_results(results, output_path)
        
        # 统计结果
        successful = sum(1 for r in results if r["success"])
        failed = len(results) - successful
        
        print(f"\nProcessing completed:")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        
        if successful > 0:
            # 统计分布
            axial_counts = {}
            manifestation_counts = {}
            
            for result in results:
                if result["success"]:
                    axial = result["axial_distribution"]
                    manifestation = result["manifestation"]
                    
                    axial_counts[axial] = axial_counts.get(axial, 0) + 1
                    manifestation_counts[manifestation] = manifestation_counts.get(manifestation, 0) + 1
            
            print(f"\nAxial distribution:")
            for key, count in axial_counts.items():
                print(f"  {key}: {count}")
            
            print(f"\nManifestation:")
            for key, count in manifestation_counts.items():
                print(f"  {key}: {count}")


if __name__ == "__main__":
    main()