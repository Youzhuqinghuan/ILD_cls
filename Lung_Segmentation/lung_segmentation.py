#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
肺部分割程序
使用阈值和形态学方法对CT图像进行肺部分割
处理 /home/huchengpeng/workspace/dataset/images 中的CT图像
"""

import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import disk, binary_erosion, binary_closing
from skimage.filters import roberts
from scipy import ndimage as ndi
from tqdm import tqdm
import argparse

def printImage(image, imgName):
    """绘制图片结果"""
    plt.figure(figsize=(10, 8))
    plt.title(imgName)
    plt.imshow(image, cmap="gray")
    plt.show()

def get_segmented_lungs(im, spacing, threshold=-300):
    """该函数用于从给定的2D切片中分割肺部"""
    # 步骤1：二值化
    binary = im < threshold
    
    # 步骤2：清除边界上的斑点
    cleared = clear_border(binary)
    
    # 步骤3：标记联通区域
    label_image = label(cleared)
    # 保留两个最大的联通区域，即左右肺部区域，其他区域全部置为0
    areas = [r.area for r in regionprops(label_image)]
    areas.sort()
    if len(areas) > 2:
        for region in regionprops(label_image):
            if region.area < areas[-2]:
                for coordinates in region.coords:
                    label_image[coordinates[0], coordinates[1]] = 0
    binary = label_image > 0
    
    # 腐蚀操作
    selem = disk(2)
    binary = binary_erosion(binary, selem)
    # 闭包操作
    selem = disk(10)
    binary = binary_closing(binary, selem)
    
    # 填充操作
    edges = roberts(binary)
    binary = ndi.binary_fill_holes(edges)
    
    # 返回最终的结果
    return binary

def process_single_ct(input_path, output_dir, visualize=False):
    """处理单个CT文件"""
    try:
        # 读取CT图像
        data = sitk.ReadImage(input_path)
        spacing = data.GetSpacing()
        scan = sitk.GetArrayFromImage(data)
        
        print(f"Processing {os.path.basename(input_path)}...")
        print(f"Image shape: {scan.shape}")
        print(f"Spacing: {spacing}")
        
        # 创建肺部掩码数组
        lung_masks = []
        segmented_scans = []
        
        # 对每个切片进行分割
        for i in tqdm(range(scan.shape[0]), desc="Processing slices"):
            slice_2d = scan[i, :, :]
            # 获取当前CT切片的mask
            mask = get_segmented_lungs(slice_2d.copy(), spacing)
            # 确保掩码是二值的：肺部区域为1，背景为0
            binary_mask = mask.astype(np.uint8)
            lung_masks.append(binary_mask)
            
            # 应用掩码
            segmented_slice = slice_2d.copy()
            segmented_slice[~mask] = 0
            segmented_scans.append(segmented_slice)
        
        # 转换为numpy数组
        lung_masks = np.array(lung_masks)
        segmented_scans = np.array(segmented_scans)
        
        # 保存结果
        base_name = os.path.basename(input_path).replace('_0000.nii.gz', '')
        
        # 保存肺部掩码
        mask_sitk = sitk.GetImageFromArray(lung_masks.astype(np.uint8))
        mask_sitk.SetSpacing(spacing)
        mask_sitk.SetOrigin(data.GetOrigin())
        mask_sitk.SetDirection(data.GetDirection())
        mask_output_path = os.path.join(output_dir, f"{base_name}_lung_mask.nii.gz")
        sitk.WriteImage(mask_sitk, mask_output_path)
        
        # 不再保存分割后的CT图像，只保存肺部掩码
        
        print(f"Saved mask: {mask_output_path}")
        
        # 可视化中间切片
        if visualize:
            mid_slice = scan.shape[0] // 2
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.title(f"Original - Slice {mid_slice}")
            plt.imshow(scan[mid_slice], cmap="gray")
            plt.axis('off')
            
            plt.subplot(1, 3, 2)
            plt.title(f"Lung Mask - Slice {mid_slice}")
            plt.imshow(lung_masks[mid_slice], cmap="gray")
            plt.axis('off')
            
            plt.subplot(1, 3, 3)
            plt.title(f"Segmented - Slice {mid_slice}")
            plt.imshow(segmented_scans[mid_slice], cmap="gray")
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{base_name}_visualization.png"), dpi=150, bbox_inches='tight')
            plt.show()
        
        return True
        
    except Exception as e:
        print(f"Error processing {input_path}: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='肺部分割程序')
    parser.add_argument('--input_dir', type=str, 
                       default='/home/huchengpeng/workspace/dataset/images',
                       help='输入CT图像目录')
    parser.add_argument('--output_dir', type=str,
                       default='/home/huchengpeng/workspace/Lung_Segmentation/results',
                       help='输出结果目录')
    parser.add_argument('--visualize', action='store_true',
                       help='是否显示可视化结果')
    parser.add_argument('--test_single', type=str, default=None,
                       help='测试单个文件，例如：CT_NSIP1_0000.nii.gz')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.test_single:
        # 测试单个文件
        input_path = os.path.join(args.input_dir, args.test_single)
        if os.path.exists(input_path):
            print(f"Testing single file: {args.test_single}")
            success = process_single_ct(input_path, args.output_dir, args.visualize)
            if success:
                print("Single file processing completed successfully!")
            else:
                print("Single file processing failed!")
        else:
            print(f"File not found: {input_path}")
        return
    
    # 处理所有CT文件
    input_dir = args.input_dir
    
    # 生成所有需要处理的文件名
    ct_files = []
    
    # UIP 1-23
    for i in range(1, 24):
        filename = f"CT_UIP{i}_0000.nii.gz"
        filepath = os.path.join(input_dir, filename)
        if os.path.exists(filepath):
            ct_files.append(filepath)
        else:
            print(f"Warning: File not found - {filename}")
    
    # NSIP 1-19
    for i in range(1, 20):
        filename = f"CT_NSIP{i}_0000.nii.gz"
        filepath = os.path.join(input_dir, filename)
        if os.path.exists(filepath):
            ct_files.append(filepath)
        else:
            print(f"Warning: File not found - {filename}")
    
    print(f"Found {len(ct_files)} CT files to process")
    
    # 处理所有文件
    successful = 0
    failed = 0
    
    for ct_file in ct_files:
        print(f"\n{'='*60}")
        success = process_single_ct(ct_file, args.output_dir, args.visualize)
        if success:
            successful += 1
        else:
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"Processing completed!")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Results saved in: {args.output_dir}")

if __name__ == "__main__":
    main()