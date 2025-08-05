import argparse
import os
import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# label_dict, 按需修改
label_dict = {
    1: 'ILD1',
    2: 'Reticulation',
    3: 'GGO-reticulation',
    4: 'ILD4',
    5: 'Honeycombing',
}

def compute_dice_coefficient(mask_gt, mask_pred):
    """计算Soerensen-Dice系数"""
    intersection = np.sum(mask_gt * mask_pred)
    return 2 * intersection / (np.sum(mask_gt) + np.sum(mask_pred))

def compute_multi_class_dsc(gt_data, pred_data):
    """
    计算多类别的 DSC（Dice 相似度系数）
    - gt_data: Ground Truth 数据，3D NumPy 数组
    - pred_data: 预测结果数据，3D NumPy 数组
    """
    dsc = {}
    
    # 遍历所有类别
    for i in label_dict.keys():
        gt_i = (gt_data == i).astype(np.uint8)  # gt 数据的二值化
        pred_i = (pred_data == i).astype(np.uint8)  # pred 数据的二值化
        
        if np.sum(gt_i) == 0 and np.sum(pred_i) == 0:
            dsc[label_dict[i]] = np.nan  # 都为空时，返回 NaN
        elif np.sum(gt_i) == 0 and np.sum(pred_i) > 0:
            dsc[label_dict[i]] = 0  # gt 为空，但预测非空时，返回 0
        else:
            dsc[label_dict[i]] = compute_dice_coefficient(gt_i, pred_i)  # 计算 DSC
    
    return dsc

def calculate_dsc_per_slice(pred_path, gt_path, label_dict):
    # 获取文件夹下的所有预测文件
    pred_files = [f for f in os.listdir(pred_path) if f.endswith('_gt.nii.gz')]
    
    # 存储每个类别的总 DSC 和切片计数
    dsc_summary = {label_dict[key]: [] for key in label_dict.keys()}
    slice_counts = {label_dict[key]: 0 for key in label_dict.keys()}  # 每个类别的有效切片计数

    # 遍历所有预测文件
    for pred_file in pred_files:
        # 预测文件名，如 CT_UIP18_gt.nii.gz
        base_name = pred_file.replace('_gt.nii.gz', '')  # 去掉 "_gt" 后缀
        
        # 对应的标签文件名应为 base_name + ".nii.gz"
        gt_file = base_name + '.nii.gz'
        
        # 确保标签文件存在
        gt_file_path = os.path.join(gt_path, gt_file)
        if not os.path.exists(gt_file_path):
            print(f"标签文件未找到: {gt_file_path}, 跳过该文件。")
            continue
        
        # 加载预测和标签数据
        pred_img = nib.load(os.path.join(pred_path, pred_file))
        gt_img = nib.load(gt_file_path)
        
        pred_data = pred_img.get_fdata()
        gt_data = gt_img.get_fdata()

        # 假设数据是 3D 形状
        assert pred_data.shape == gt_data.shape, f"Shape mismatch for {pred_file} and {gt_file}"

        # 遍历每一层切片
        for slice_idx in range(pred_data.shape[2]):  # 按 z 轴切片
            pred_slice = pred_data[:, :, slice_idx]
            gt_slice = gt_data[:, :, slice_idx]
            
            # 计算每个类别的 DSC
            dsc = compute_multi_class_dsc(gt_slice, pred_slice)
            
            # 将每个切片的 DSC 添加到统计中
            for label, value in dsc.items():
                dsc_summary[label].append(value)
                if not np.isnan(value):  # 统计有效切片数量
                    slice_counts[label] += 1
    
    # 计算每个类别的平均 DSC
    mean_dsc = {label: np.nanmean(dsc_values) for label, dsc_values in dsc_summary.items()}
    
    return mean_dsc, dsc_summary, slice_counts


# 添加 argparse 解析命令行参数
parser = argparse.ArgumentParser()

parser.add_argument('-s', '--seg_dir', default=None, type=str, required=True, help='目录路径，存储预测结果的nii.gz文件')
parser.add_argument('-g', '--gt_dir',  default=None, type=str, required=True, help='目录路径，存储标签文件的nii.gz文件')
parser.add_argument('-csv_dir', default='./', type=str, help='保存计算结果的CSV文件夹')
args = parser.parse_args()

# 获取命令行参数
seg_dir = args.seg_dir
gt_dir = args.gt_dir
csv_dir = args.csv_dir
os.makedirs(csv_dir, exist_ok=True)  # 创建目录

# 调用计算 DSC 函数
mean_dsc, dsc_summary, slice_counts = calculate_dsc_per_slice(seg_dir, gt_dir, label_dict)

# 打印每个类别的平均 DSC
for label, dsc in mean_dsc.items():
    print(f"{label}: {dsc:.4f} (Total slices: {slice_counts[label]})")

# 保存到 CSV 文件
df = pd.DataFrame(mean_dsc.items(), columns=['Label', 'Mean DSC'])
df.to_csv(os.path.join(csv_dir, 'mean_dsc.csv'), index=False)

# 为每个类别绘制柱状图
for label, dsc_values in dsc_summary.items():
    valid_dsc_values = [dsc for dsc in dsc_values if not np.isnan(dsc)]  # 忽略 NaN 值
    valid_indices = [idx for idx, dsc in enumerate(dsc_values) if not np.isnan(dsc)]
    
    plt.figure(figsize=(10, 6))
    plt.bar(valid_indices, valid_dsc_values, color='skyblue')
    plt.xlabel('Slice Index')
    plt.ylabel('DSC')
    plt.title(f'DSC for {label}')
    
    # 保存图表
    plt.savefig(os.path.join(csv_dir, f'DSC_{label}.png'))
    plt.close()
