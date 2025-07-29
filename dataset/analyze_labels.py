#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
标签文件分析程序
读取labels文件夹下的nii.gz文件，统计每个slice中不同标签的出现次数
"""

import os
import numpy as np
import nibabel as nib
from collections import defaultdict
import pandas as pd
from pathlib import Path

def analyze_nii_labels(labels_dir):
    """
    分析labels文件夹中所有nii.gz文件的标签分布
    
    Args:
        labels_dir (str): labels文件夹路径
    
    Returns:
        dict: 包含每个文件每个slice标签统计的字典
    """
    labels_path = Path(labels_dir)
    
    if not labels_path.exists():
        print(f"错误: 文件夹 {labels_dir} 不存在")
        return {}
    
    # 查找所有nii.gz文件
    nii_files = list(labels_path.glob('*.nii.gz'))
    
    if not nii_files:
        print(f"警告: 在 {labels_dir} 中未找到nii.gz文件")
        return {}
    
    print(f"找到 {len(nii_files)} 个nii.gz文件")
    
    results = {}
    
    for nii_file in nii_files:
        print(f"\n处理文件: {nii_file.name}")
        
        try:
            # 加载nii.gz文件
            img = nib.load(str(nii_file))
            data = img.get_fdata()
            
            print(f"  文件形状: {data.shape}")
            
            # 获取文件名（不含扩展名）
            file_key = nii_file.stem.replace('.nii', '')
            results[file_key] = {}
            
            # 遍历每个slice（假设第三个维度是slice）
            if len(data.shape) == 3:
                num_slices = data.shape[2]
                
                for slice_idx in range(num_slices):
                    slice_data = data[:, :, slice_idx]
                    
                    # 统计每个标签的出现次数
                    unique_labels, counts = np.unique(slice_data, return_counts=True)
                    
                    # 完全去除标签值为0的统计
                    non_zero_mask = unique_labels != 0
                    unique_labels = unique_labels[non_zero_mask]
                    counts = counts[non_zero_mask]
                    
                    # 存储结果
                    slice_stats = {}
                    for label, count in zip(unique_labels, counts):
                        slice_stats[int(label)] = int(count)
                    
                    results[file_key][f'slice_{slice_idx:03d}'] = slice_stats
                    
                    # 打印非空slice的统计信息
                    if slice_stats:
                        print(f"    Slice {slice_idx:03d}: {slice_stats}")
            
            elif len(data.shape) == 2:
                # 2D图像，只有一个slice
                unique_labels, counts = np.unique(data, return_counts=True)
                
                # 完全去除标签值为0的统计
                non_zero_mask = unique_labels != 0
                unique_labels = unique_labels[non_zero_mask]
                counts = counts[non_zero_mask]
                
                slice_stats = {}
                for label, count in zip(unique_labels, counts):
                    slice_stats[int(label)] = int(count)
                
                results[file_key]['slice_000'] = slice_stats
                print(f"    单个slice: {slice_stats}")
            
            else:
                print(f"  警告: 不支持的数据维度 {data.shape}")
                
        except Exception as e:
            print(f"  错误: 处理文件 {nii_file.name} 时出错: {str(e)}")
            continue
    
    return results

def save_results_to_csv(results, output_file):
    """
    将结果保存为CSV文件
    
    Args:
        results (dict): 分析结果
        output_file (str): 输出文件路径
    """
    rows = []
    
    for file_name, file_data in results.items():
        for slice_name, slice_data in file_data.items():
            for label, count in slice_data.items():
                rows.append({
                    'file_name': file_name,
                    'slice': slice_name,
                    'label': label,
                    'count': count
                })
    
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"\n结果已保存到: {output_file}")
    else:
        print("\n没有数据可保存")

def print_summary(results):
    """
    打印统计摘要
    
    Args:
        results (dict): 分析结果
    """
    print("\n=== 统计摘要 ===")
    
    total_files = len(results)
    total_slices = sum(len(file_data) for file_data in results.values())
    
    # 统计所有出现的标签
    all_labels = set()
    for file_data in results.values():
        for slice_data in file_data.values():
            all_labels.update(slice_data.keys())
    
    print(f"处理的文件数: {total_files}")
    print(f"总slice数: {total_slices}")
    print(f"发现的标签: {sorted(all_labels)}")
    
    # 每个标签的总计数和文件分布
    label_totals = defaultdict(int)
    label_files = defaultdict(set)
    
    for file_name, file_data in results.items():
        for slice_data in file_data.values():
            for label, count in slice_data.items():
                label_totals[label] += count
                label_files[label].add(file_name)
    
    print("\n各标签总计数:")
    for label in sorted(label_totals.keys()):
        print(f"  标签 {label}: {label_totals[label]}")
    
    print("\n各标签出现的文件:")
    for label in sorted(label_files.keys()):
        files_list = sorted(list(label_files[label]))
        print(f"  标签 {label}: {files_list}")

def main():
    """
    主函数
    """
    # 设置路径
    current_dir = Path(__file__).parent
    labels_dir = current_dir / 'labels'
    output_file = current_dir / 'label_statistics.csv'
    
    print("开始分析nii.gz文件中的标签分布...")
    print(f"标签文件夹: {labels_dir}")
    
    # 分析标签文件
    results = analyze_nii_labels(str(labels_dir))
    
    if results:
        # 打印摘要
        print_summary(results)
        
        # 保存结果
        save_results_to_csv(results, str(output_file))
        
        print("\n分析完成!")
    else:
        print("\n没有找到可分析的文件或所有文件处理失败")

if __name__ == '__main__':
    main()