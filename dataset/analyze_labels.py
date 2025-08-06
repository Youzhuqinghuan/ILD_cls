#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
标签文件分析程序
以slice为单位统计每个患者的病变标签，以患者为单位存储存在病变的roi_layer和每张slice的病变标签
"""

import os
import numpy as np
import nibabel as nib
import json
from collections import defaultdict
from pathlib import Path

def analyze_patient_labels(labels_dir):
    """
    以患者为单位分析labels文件夹中所有nii.gz文件的病变标签分布
    
    Args:
        labels_dir (str): labels文件夹路径
    
    Returns:
        dict: 包含每个患者病变信息的字典，格式为:
        {
            "patient_id": {
                "roi_layers_with_lesions": [list of slice indices with lesions],
                "slice_lesion_labels": {
                    "slice_xxx": [list of lesion labels in this slice]
                }
            }
        }
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
    
    patient_results = {}
    
    for nii_file in nii_files:
        print(f"\n处理文件: {nii_file.name}")
        
        try:
            # 加载nii.gz文件
            img = nib.load(str(nii_file))
            data = img.get_fdata()
            
            print(f"  文件形状: {data.shape}")
            
            # 获取患者ID（文件名不含扩展名）
            patient_id = nii_file.stem.replace('.nii', '')
            
            # 初始化患者数据结构
            patient_results[patient_id] = {
                "roi_layers_with_lesions": [],
                "slice_lesion_labels": {}
            }
            
            # 遍历每个slice（假设第三个维度是slice）
            if len(data.shape) == 3:
                num_slices = data.shape[2]
                
                for slice_idx in range(num_slices):
                    slice_data = data[:, :, slice_idx]
                    
                    # 获取该slice中的所有非零标签（病变标签）
                    unique_labels = np.unique(slice_data)
                    lesion_labels = [int(label) for label in unique_labels if label != 0]
                    
                    # 如果该slice存在病变标签
                    if lesion_labels:
                        slice_name = f'slice_{slice_idx:03d}'
                        patient_results[patient_id]["roi_layers_with_lesions"].append(slice_idx)
                        patient_results[patient_id]["slice_lesion_labels"][slice_name] = lesion_labels
                        
                        print(f"    Slice {slice_idx:03d}: 病变标签 {lesion_labels}")
            
            elif len(data.shape) == 2:
                # 2D图像，只有一个slice
                unique_labels = np.unique(data)
                lesion_labels = [int(label) for label in unique_labels if label != 0]
                
                if lesion_labels:
                    patient_results[patient_id]["roi_layers_with_lesions"].append(0)
                    patient_results[patient_id]["slice_lesion_labels"]["slice_000"] = lesion_labels
                    print(f"    单个slice: 病变标签 {lesion_labels}")
            
            else:
                print(f"  警告: 不支持的数据维度 {data.shape}")
                
        except Exception as e:
            print(f"  错误: 处理文件 {nii_file.name} 时出错: {str(e)}")
            continue
    
    return patient_results

def save_results_to_json(results, output_file):
    """
    将结果保存为JSON文件
    
    Args:
        results (dict): 分析结果
        output_file (str): 输出文件路径
    """
    if results:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
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
    
    total_patients = len(results)
    total_slices_with_lesions = sum(len(patient_data["roi_layers_with_lesions"]) for patient_data in results.values())
    
    # 统计所有出现的病变标签
    all_labels = set()
    for patient_data in results.values():
        for slice_labels in patient_data["slice_lesion_labels"].values():
            all_labels.update(slice_labels)
    
    print(f"处理的患者数: {total_patients}")
    print(f"存在病变的slice总数: {total_slices_with_lesions}")
    print(f"发现的病变标签: {sorted(all_labels)}")
    
    # 每个标签出现的患者数和slice数
    label_patients = defaultdict(set)
    label_slice_count = defaultdict(int)
    
    for patient_id, patient_data in results.items():
        for slice_labels in patient_data["slice_lesion_labels"].values():
            for label in slice_labels:
                label_patients[label].add(patient_id)
                label_slice_count[label] += 1
    
    print("\n各病变标签统计:")
    for label in sorted(all_labels):
        patient_count = len(label_patients[label])
        slice_count = label_slice_count[label]
        print(f"  标签 {label}: 出现在 {patient_count} 个患者的 {slice_count} 个slice中")
    
    print("\n各患者病变分布:")
    for patient_id, patient_data in results.items():
        lesion_slices = len(patient_data["roi_layers_with_lesions"])
        all_patient_labels = set()
        for slice_labels in patient_data["slice_lesion_labels"].values():
            all_patient_labels.update(slice_labels)
        print(f"  患者 {patient_id}: {lesion_slices} 个病变slice，病变标签: {sorted(all_patient_labels)}")

def main():
    """
    主函数
    """
    # 设置路径
    current_dir = Path(__file__).parent
    labels_dir = current_dir / 'labels'
    output_file = current_dir / 'patient_lesion_analysis.json'
    
    print("开始分析患者病变标签分布...")
    print(f"标签文件夹: {labels_dir}")
    
    # 分析标签文件
    results = analyze_patient_labels(str(labels_dir))
    
    if results:
        # 打印摘要
        print_summary(results)
        
        # 保存结果
        save_results_to_json(results, str(output_file))
        
        print("\n分析完成!")
    else:
        print("\n没有找到可分析的文件或所有文件处理失败")

if __name__ == '__main__':
    main()