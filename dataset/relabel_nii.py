#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
标签重新映射程序
将nii.gz文件中的标签按指定规则重新映射：
- 标签3 → 标签2
- 标签4 → 标签3  
- 标签5 → 标签4
"""

import os
import numpy as np
import nibabel as nib
from pathlib import Path
import shutil
from datetime import datetime

def relabel_nii_file(input_file, output_file, label_mapping, backup_dir=None):
    """
    重新映射单个nii.gz文件的标签
    
    Args:
        input_file (str): 输入文件路径
        output_file (str): 输出文件路径
        label_mapping (dict): 标签映射字典 {原标签: 新标签}
        backup_dir (str): 备份目录路径
    
    Returns:
        dict: 转换统计信息
    """
    try:
        # 创建备份
        if backup_dir and input_file == output_file:
            backup_path = Path(backup_dir) / Path(input_file).name
            shutil.copy2(input_file, backup_path)
            print(f"  已备份原文件到: {backup_path}")
        
        # 加载nii.gz文件
        img = nib.load(input_file)
        data = img.get_fdata().astype(np.int32)  # 确保是整数类型
        original_data = data.copy()
        
        print(f"  文件形状: {data.shape}")
        print(f"  原始标签值: {sorted(np.unique(data))}")
        
        # 统计转换前的标签分布
        original_counts = {}
        for label in label_mapping.keys():
            count = np.sum(data == label)
            if count > 0:
                original_counts[label] = count
                print(f"    标签 {label}: {count} 个像素")
        
        # 使用临时值进行标签转换，避免冲突
        # 第一步：将需要转换的标签先映射到临时值（负数）
        temp_mapping = {}
        for old_label, new_label in label_mapping.items():
            temp_value = -(old_label + 1000)  # 使用负数作为临时值
            temp_mapping[old_label] = temp_value
            data[original_data == old_label] = temp_value
        
        # 第二步：将临时值映射到最终值
        for old_label, new_label in label_mapping.items():
            temp_value = temp_mapping[old_label]
            data[data == temp_value] = new_label
        
        print(f"  转换后标签值: {sorted(np.unique(data))}")
        
        # 统计转换后的标签分布
        converted_counts = {}
        for new_label in label_mapping.values():
            count = np.sum(data == new_label)
            if count > 0:
                converted_counts[new_label] = count
                print(f"    标签 {new_label}: {count} 个像素")
        
        # 创建新的nii图像并保存
        new_img = nib.Nifti1Image(data, img.affine, img.header)
        nib.save(new_img, output_file)
        
        return {
            'success': True,
            'original_counts': original_counts,
            'converted_counts': converted_counts,
            'total_converted_pixels': sum(original_counts.values())
        }
        
    except Exception as e:
        print(f"  错误: {str(e)}")
        return {'success': False, 'error': str(e)}

def relabel_all_files(labels_dir, output_dir=None, create_backup=True):
    """
    批量重新映射labels文件夹中的所有nii.gz文件
    
    Args:
        labels_dir (str): 标签文件夹路径
        output_dir (str): 输出文件夹路径，如果为None则覆盖原文件
        create_backup (bool): 是否创建备份
    
    Returns:
        dict: 处理结果统计
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
    
    # 设置输出目录
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"输出目录: {output_path}")
    else:
        output_path = labels_path
        print("将覆盖原文件")
    
    # 设置备份目录
    backup_dir = None
    if create_backup and not output_dir:  # 只有覆盖原文件时才需要备份
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = labels_path / f"backup_{timestamp}"
        backup_dir.mkdir(exist_ok=True)
        print(f"备份目录: {backup_dir}")
    
    # 定义标签映射规则
    label_mapping = {
        3: 2,  # 标签3 → 标签2
        4: 3,  # 标签4 → 标签3
        5: 4   # 标签5 → 标签4
    }
    
    print(f"\n标签映射规则: {label_mapping}")
    print("开始处理文件...\n")
    
    results = {}
    total_files = len(nii_files)
    successful_files = 0
    total_converted_pixels = 0
    
    for i, nii_file in enumerate(nii_files, 1):
        print(f"[{i}/{total_files}] 处理文件: {nii_file.name}")
        
        # 确定输出文件路径
        if output_dir:
            output_file = output_path / nii_file.name
        else:
            output_file = nii_file
        
        # 处理文件
        result = relabel_nii_file(
            str(nii_file), 
            str(output_file), 
            label_mapping, 
            str(backup_dir) if backup_dir else None
        )
        
        results[nii_file.name] = result
        
        if result['success']:
            successful_files += 1
            total_converted_pixels += result.get('total_converted_pixels', 0)
            print(f"  ✓ 处理成功")
        else:
            print(f"  ✗ 处理失败: {result.get('error', '未知错误')}")
        
        print()
    
    # 打印总结
    print("=== 处理总结 ===")
    print(f"总文件数: {total_files}")
    print(f"成功处理: {successful_files}")
    print(f"失败文件: {total_files - successful_files}")
    print(f"总转换像素数: {total_converted_pixels}")
    
    if backup_dir and backup_dir.exists():
        print(f"备份位置: {backup_dir}")
    
    return results

def main():
    """
    主函数
    """
    # 设置路径
    current_dir = Path(__file__).parent
    labels_dir = current_dir / 'labels'
    
    print("标签重新映射程序")
    print("映射规则: 3→2, 4→3, 5→4")
    print(f"处理目录: {labels_dir}")
    print()
    
    # 询问用户选择
    print("请选择处理方式:")
    print("1. 覆盖原文件（会自动创建备份）")
    print("2. 创建新文件到指定目录")
    
    try:
        choice = input("请输入选择 (1 或 2): ").strip()
        
        if choice == '1':
            # 覆盖原文件
            results = relabel_all_files(str(labels_dir), output_dir=None, create_backup=True)
        elif choice == '2':
            # 创建新文件
            output_dir = input("请输入输出目录路径: ").strip()
            if not output_dir:
                output_dir = str(current_dir / 'labels_relabeled')
                print(f"使用默认输出目录: {output_dir}")
            results = relabel_all_files(str(labels_dir), output_dir=output_dir, create_backup=False)
        else:
            print("无效选择，程序退出")
            return
        
        if results:
            print("\n处理完成!")
        else:
            print("\n没有文件被处理")
            
    except KeyboardInterrupt:
        print("\n用户取消操作")
    except Exception as e:
        print(f"\n程序执行出错: {str(e)}")

if __name__ == '__main__':
    main()