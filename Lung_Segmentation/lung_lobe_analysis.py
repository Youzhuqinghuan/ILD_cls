#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
肺叶分析程序
实现上下肺划分和病变占比计算
基于几何百分位数方法划分上下肺
"""

import os
import numpy as np
import SimpleITK as sitk
import pandas as pd
import json
from datetime import datetime
import argparse

def load_nifti_data(file_path):
    """加载NIfTI文件数据"""
    try:
        data = sitk.ReadImage(file_path)
        array = sitk.GetArrayFromImage(data)
        spacing = data.GetSpacing()
        return array, spacing, data
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        return None, None, None

def divide_upper_lower_lung(lung_mask, percentile=60):
    """使用几何百分位数方法划分上下肺
    
    Args:
        lung_mask: 二值化肺部掩码 (0和1)
        percentile: 百分位数，默认60%作为分界点
    
    Returns:
        upper_lung: 上肺掩码
        lower_lung: 下肺掩码
        z_cut: 分界切片位置
    """
    # 获取所有肺部像素的z坐标
    z_vox = np.where(lung_mask)[0]  # 第一个维度是z轴
    
    if len(z_vox) == 0:
        print("Warning: No lung pixels found in mask")
        return None, None, None
    
    # 计算分界点
    z_cut = int(np.percentile(z_vox, percentile))
    
    # 创建上下肺掩码
    upper_lung = lung_mask.copy()
    lower_lung = lung_mask.copy()
    
    # 上肺：保留z_cut以上的部分（CT图像中z轴较大的是上方/头部方向）
    upper_lung[:z_cut, :, :] = 0

    # 下肺：保留z_cut以下的部分
    lower_lung[z_cut:, :, :] = 0
    
    return upper_lung, lower_lung, z_cut

def calculate_volume(mask, spacing):
    """计算体积（考虑像素间距）
    
    Args:
        mask: 二值掩码
        spacing: 像素间距 (x, y, z)
    
    Returns:
        volume: 体积（立方毫米）
    """
    if mask is None:
        return 0
    
    # 计算像素数量
    pixel_count = np.sum(mask > 0)
    
    # 计算单个体素的体积（立方毫米）
    voxel_volume = spacing[0] * spacing[1] * spacing[2]
    
    # 总体积
    volume = pixel_count * voxel_volume
    
    return volume

def analyze_lesion_distribution(lung_mask_path, lesion_path, case_name):
    """分析病变在上下肺的分布
    
    Args:
        lung_mask_path: 肺部掩码文件路径
        lesion_path: 病变分割文件路径
        case_name: 病例名称
    
    Returns:
        result: 分析结果字典
    """
    result = {
        'case_name': case_name,
        'lung_mask_path': lung_mask_path,
        'lesion_path': lesion_path,
        'success': False,
        'error': None
    }
    
    try:
        # 加载肺部掩码
        lung_mask, spacing, lung_data = load_nifti_data(lung_mask_path)
        if lung_mask is None:
            result['error'] = f"Failed to load lung mask: {lung_mask_path}"
            return result
        
        # 加载病变分割
        lesion_mask, _, _ = load_nifti_data(lesion_path)
        if lesion_mask is None:
            result['error'] = f"Failed to load lesion mask: {lesion_path}"
            return result
        
        # 确保掩码形状一致
        if lung_mask.shape != lesion_mask.shape:
            result['error'] = f"Shape mismatch: lung {lung_mask.shape} vs lesion {lesion_mask.shape}"
            return result
        
        print(f"\nAnalyzing {case_name}...")
        print(f"Lung mask shape: {lung_mask.shape}")
        print(f"Lesion mask shape: {lesion_mask.shape}")
        print(f"Spacing: {spacing}")
        
        # 二值化处理（确保是0和1）
        lung_mask = (lung_mask > 0).astype(np.uint8)
        lesion_mask = (lesion_mask > 0).astype(np.uint8)
        
        # 划分上下肺
        upper_lung, lower_lung, z_cut = divide_upper_lower_lung(lung_mask, percentile=60)
        
        if upper_lung is None:
            result['error'] = "Failed to divide upper/lower lung"
            return result
        
        print(f"Z-axis cut position: {z_cut}")
        
        # 计算肺部体积
        total_lung_volume = calculate_volume(lung_mask, spacing)
        upper_lung_volume = calculate_volume(upper_lung, spacing)
        lower_lung_volume = calculate_volume(lower_lung, spacing)
        
        # 计算病变在上下肺的分布
        lesion_in_upper = lesion_mask * upper_lung
        lesion_in_lower = lesion_mask * lower_lung
        
        # 计算病变体积
        total_lesion_volume = calculate_volume(lesion_mask, spacing)
        upper_lesion_volume = calculate_volume(lesion_in_upper, spacing)
        lower_lesion_volume = calculate_volume(lesion_in_lower, spacing)
        
        # 计算占比
        upper_lesion_ratio = (upper_lesion_volume / upper_lung_volume * 100) if upper_lung_volume > 0 else 0
        lower_lesion_ratio = (lower_lesion_volume / lower_lung_volume * 100) if lower_lung_volume > 0 else 0
        total_lesion_ratio = (total_lesion_volume / total_lung_volume * 100) if total_lung_volume > 0 else 0
        
        # 病变在上下肺的分布比例
        upper_lesion_proportion = (upper_lesion_volume / total_lesion_volume * 100) if total_lesion_volume > 0 else 0
        lower_lesion_proportion = (lower_lesion_volume / total_lesion_volume * 100) if total_lesion_volume > 0 else 0
        
        # 保存结果
        result.update({
            'success': True,
            'z_cut_position': int(z_cut),
            'total_slices': int(lung_mask.shape[0]),
            'spacing': list(spacing),
            
            # 肺部体积信息（立方毫米）
            'total_lung_volume_mm3': float(total_lung_volume),
            'upper_lung_volume_mm3': float(upper_lung_volume),
            'lower_lung_volume_mm3': float(lower_lung_volume),
            
            # 病变体积信息（立方毫米）
            'total_lesion_volume_mm3': float(total_lesion_volume),
            'upper_lesion_volume_mm3': float(upper_lesion_volume),
            'lower_lesion_volume_mm3': float(lower_lesion_volume),
            
            # 病变在各肺叶的占比（%）
            'upper_lesion_ratio_percent': float(upper_lesion_ratio),
            'lower_lesion_ratio_percent': float(lower_lesion_ratio),
            'total_lesion_ratio_percent': float(total_lesion_ratio),
            
            # 病变在上下肺的分布比例（%）
            'upper_lesion_proportion_percent': float(upper_lesion_proportion),
            'lower_lesion_proportion_percent': float(lower_lesion_proportion),
            
            # 像素统计
            'total_lung_pixels': int(np.sum(lung_mask > 0)),
            'upper_lung_pixels': int(np.sum(upper_lung > 0)),
            'lower_lung_pixels': int(np.sum(lower_lung > 0)),
            'total_lesion_pixels': int(np.sum(lesion_mask > 0)),
            'upper_lesion_pixels': int(np.sum(lesion_in_upper > 0)),
            'lower_lesion_pixels': int(np.sum(lesion_in_lower > 0))
        })
        
        # 打印结果
        print(f"\n=== Analysis Results for {case_name} ===")
        print(f"Total lung volume: {total_lung_volume:.2f} mm³")
        print(f"Upper lung volume: {upper_lung_volume:.2f} mm³")
        print(f"Lower lung volume: {lower_lung_volume:.2f} mm³")
        print(f"\nTotal lesion volume: {total_lesion_volume:.2f} mm³")
        print(f"Upper lesion volume: {upper_lesion_volume:.2f} mm³")
        print(f"Lower lesion volume: {lower_lesion_volume:.2f} mm³")
        print(f"\nLesion ratio in upper lung: {upper_lesion_ratio:.2f}%")
        print(f"Lesion ratio in lower lung: {lower_lesion_ratio:.2f}%")
        print(f"Total lesion ratio: {total_lesion_ratio:.2f}%")
        print(f"\nLesion distribution - Upper: {upper_lesion_proportion:.2f}%, Lower: {lower_lesion_proportion:.2f}%")
        
    except Exception as e:
        result['error'] = str(e)
        print(f"Error analyzing {case_name}: {str(e)}")
    
    return result

def find_matching_files(lung_mask_dir, lesion_dir):
    """查找匹配的肺部掩码和病变分割文件"""
    matches = []
    
    # 获取所有肺部掩码文件
    lung_files = [f for f in os.listdir(lung_mask_dir) if f.endswith('_lung_mask.nii.gz')]
    
    for lung_file in lung_files:
        # 提取病例名称，例如从 "CT_NSIP1_lung_mask.nii.gz" 提取 "CT_NSIP1"
        case_name = lung_file.replace('_lung_mask.nii.gz', '')
        
        # 查找对应的病变文件
        lesion_file = f"{case_name}.nii.gz"
        lesion_path = os.path.join(lesion_dir, lesion_file)
        
        if os.path.exists(lesion_path):
            lung_path = os.path.join(lung_mask_dir, lung_file)
            matches.append((case_name, lung_path, lesion_path))
            print(f"Found match: {case_name}")
        else:
            print(f"No lesion file found for {case_name}: {lesion_file}")
    
    return matches

def save_results(results, output_dir):
    """保存分析结果为简化的JSONL格式"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存简化的JSONL结果
    jsonl_file = os.path.join(output_dir, f"lung_lobe_results_{timestamp}.jsonl")
    
    with open(jsonl_file, 'w', encoding='utf-8') as f:
        for result in results:
            if result['success']:
                # 只保留关键信息
                simplified_result = {
                    'filename': result['case_name'],
                    'upper_lesion_ratio': result['upper_lesion_ratio_percent'],
                    'lower_lesion_ratio': result['lower_lesion_ratio_percent'],
                    'total_lesion_ratio': result['total_lesion_ratio_percent']
                }
                f.write(json.dumps(simplified_result, ensure_ascii=False) + '\n')
    
    print(f"\nResults saved to: {jsonl_file}")
    return jsonl_file

def main():
    parser = argparse.ArgumentParser(description='肺叶分析程序 - 上下肺划分和病变占比计算')
    parser.add_argument('--lung_mask_dir', type=str,
                       default='/home/huchengpeng/workspace/Lung_Segmentation/results',
                       help='肺部掩码文件目录')
    parser.add_argument('--lesion_dir', type=str,
                       default='/home/huchengpeng/workspace/dataset/labels',
                       help='病变分割文件目录')
    parser.add_argument('--output_dir', type=str,
                       default='/home/huchengpeng/workspace/Lung_Segmentation/analysis_results',
                       help='输出结果目录')
    parser.add_argument('--test_single', type=str, default=None,
                       help='测试单个病例，例如：CT_NSIP1')
    parser.add_argument('--percentile', type=int, default=60,
                       help='上下肺划分的百分位数（默认60%）')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    results = []
    
    if args.test_single:
        # 测试单个病例
        case_name = args.test_single
        lung_mask_path = os.path.join(args.lung_mask_dir, f"{case_name}_lung_mask.nii.gz")
        lesion_path = os.path.join(args.lesion_dir, f"{case_name}.nii.gz")
        
        if not os.path.exists(lung_mask_path):
            print(f"Lung mask not found: {lung_mask_path}")
            return
        
        if not os.path.exists(lesion_path):
            print(f"Lesion file not found: {lesion_path}")
            return
        
        print(f"Testing single case: {case_name}")
        result = analyze_lesion_distribution(lung_mask_path, lesion_path, case_name)
        results.append(result)
    else:
        # 批量处理
        matches = find_matching_files(args.lung_mask_dir, args.lesion_dir)
        
        if not matches:
            print("No matching files found!")
            return
        
        print(f"Found {len(matches)} matching cases")
        
        for case_name, lung_path, lesion_path in matches:
            result = analyze_lesion_distribution(lung_path, lesion_path, case_name)
            results.append(result)
    
    # 保存结果
    if results:
        jsonl_file = save_results(results, args.output_dir)
        
        # 统计成功和失败的数量
        successful = sum(1 for r in results if r['success'])
        failed = len(results) - successful
        
        print(f"\n=== Summary ===")
        print(f"Total cases processed: {len(results)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        
        if failed > 0:
            print("\nFailed cases:")
            for result in results:
                if not result['success']:
                    print(f"  {result['case_name']}: {result['error']}")

if __name__ == "__main__":
    main()