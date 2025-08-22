# -*- coding: utf-8 -*-
# %% import packages
# pip install connected-components-3d
import numpy as np

# import nibabel as nib
import SimpleITK as sitk
import os

join = os.path.join
from skimage import transform
from tqdm import tqdm
import cc3d

# convert nii image to npz files, including original image and corresponding masks
modality = "CT"
anatomy = "ILD"  # anantomy + dataset name
img_name_suffix = "_0000.nii.gz"
gt_name_suffix = ".nii.gz"
lung_name_suffix = "_lung_mask.nii.gz"

# Processing mode control
# "all": process imgs, gts and lungs
# "lungs_only": only process lung masks
processing_mode = "lungs_only"  # Change this to "all" to process all data types

nii_path = "/home/huchengpeng/workspace/ILD_cls/dataset/images"  # path to the nii images
gt_path = "/home/huchengpeng/workspace/ILD_cls/dataset/labels"  # path to the ground truth
lung_path = "/home/huchengpeng/workspace/ILD_cls/dataset/lungs"  # path to the lung masks
npy_path = "./ILD/data/rawdata"
os.makedirs(join(npy_path, "gts"), exist_ok=True)
os.makedirs(join(npy_path, "imgs"), exist_ok=True)
os.makedirs(join(npy_path, "lungs"), exist_ok=True)

image_size = 1024
voxel_num_thre2d = 100
voxel_num_thre3d = 1000

if processing_mode == "lungs_only":
    # For lungs_only mode, use gt files as reference but only process lungs
    names = sorted(os.listdir(gt_path))
    print(f"ori # files {len(names)=}")
    names = [
        name
        for name in names
        if os.path.exists(join(lung_path, name.split(gt_name_suffix)[0] + lung_name_suffix))
    ]
    print(f"after sanity check # files {len(names)=}")
else:
    # For all mode, check all required files exist
    names = sorted(os.listdir(gt_path))
    print(f"ori # files {len(names)=}")
    names = [
        name
        for name in names
        if os.path.exists(join(nii_path, name.split(gt_name_suffix)[0] + img_name_suffix))
        and os.path.exists(join(lung_path, name.split(gt_name_suffix)[0] + lung_name_suffix))
    ]
    print(f"after sanity check # files {len(names)=}")

# set label ids that are excluded - removed as per requirements
# set window level and width
# https://radiopaedia.org/articles/windowing-ct
WINDOW_LEVEL = 40  # only for CT images
WINDOW_WIDTH = 400  # only for CT images

# %% save preprocessed images and masks as npy files
if processing_mode == "lungs_only":
    # Process only lung masks but use gt to determine slices
    for name in tqdm(names):  # process all available files
        base_name = name.split(gt_name_suffix)[0]
        gt_name = name
        
        # Load gt data to determine valid slices
        gt_sitk = sitk.ReadImage(join(gt_path, gt_name))
        gt_data_ori = np.uint8(sitk.GetArrayFromImage(gt_sitk))
        
        # Load lung mask data
        lung_name = base_name + lung_name_suffix
        lung_sitk = sitk.ReadImage(join(lung_path, lung_name))
        lung_data_ori = np.uint8(sitk.GetArrayFromImage(lung_sitk))
        
        # exclude the objects with less than 1000 pixels in 3D for gt
        gt_data_ori = cc3d.dust(
            gt_data_ori, threshold=voxel_num_thre3d, connectivity=26, in_place=True
        )
        # remove small objects with less than 100 pixels in 2D slices for gt
        for slice_i in range(gt_data_ori.shape[0]):
            gt_i = gt_data_ori[slice_i, :, :]
            gt_data_ori[slice_i, :, :] = cc3d.dust(
                gt_i, threshold=voxel_num_thre2d, connectivity=8, in_place=True
            )
        
        # find non-zero slices using gt data
        z_index, _, _ = np.where(gt_data_ori > 0)
        z_index = np.unique(z_index)
        
        if len(z_index) > 0:
            # crop the lung data with non-zero slices determined by gt
            lung_roi = lung_data_ori[z_index, :, :]
            
            # save each slice as npy file
            for i in range(lung_roi.shape[0]):
                # Process lung mask
                lung_i = lung_roi[i, :, :]
                resize_lung_skimg = transform.resize(
                    lung_i,
                    (image_size, image_size),
                    order=0,
                    preserve_range=True,
                    mode="constant",
                    anti_aliasing=False,
                )
                resize_lung_skimg = np.uint8(resize_lung_skimg)
                
                np.save(
                    join(
                        npy_path,
                        "lungs",
                        base_name + "-" + str(i).zfill(3) + ".npy",
                    ),
                    resize_lung_skimg,
                )
else:
    # Process all data types (imgs, gts, lungs)
    for name in tqdm(names):  # process all available files
        base_name = name.split(gt_name_suffix)[0]
        image_name = base_name + img_name_suffix
        gt_name = name
        
        gt_sitk = sitk.ReadImage(join(gt_path, gt_name))
        gt_data_ori = np.uint8(sitk.GetArrayFromImage(gt_sitk))
        
        img_sitk = sitk.ReadImage(join(nii_path, image_name))
        image_data = sitk.GetArrayFromImage(img_sitk)
        
        # Load lung mask data
        lung_name = base_name + lung_name_suffix
        lung_sitk = sitk.ReadImage(join(lung_path, lung_name))
        lung_data_ori = np.uint8(sitk.GetArrayFromImage(lung_sitk))

        # exclude the objects with less than 1000 pixels in 3D
        gt_data_ori = cc3d.dust(
            gt_data_ori, threshold=voxel_num_thre3d, connectivity=26, in_place=True
        )
        # remove small objects with less than 100 pixels in 2D slices

        for slice_i in range(gt_data_ori.shape[0]):
            gt_i = gt_data_ori[slice_i, :, :]
            # remove small objects with less than 100 pixels
            # reason: fro such small objects, the main challenge is detection rather than segmentation
            gt_data_ori[slice_i, :, :] = cc3d.dust(
                gt_i, threshold=voxel_num_thre2d, connectivity=8, in_place=True
            )
        
        # find non-zero slices
        z_index, _, _ = np.where(gt_data_ori > 0)
        z_index = np.unique(z_index)

        if len(z_index) > 0:
            # crop the gt data with non-zero slices
            gt_roi = gt_data_ori[z_index, :, :]
            # crop the lung data with the same slices
            lung_roi = lung_data_ori[z_index, :, :]
            
            # nii preprocess start
            if modality == "CT":
                lower_bound = WINDOW_LEVEL - WINDOW_WIDTH / 2
                upper_bound = WINDOW_LEVEL + WINDOW_WIDTH / 2
                image_data_pre = np.clip(image_data, lower_bound, upper_bound)
                image_data_pre = (
                    (image_data_pre - np.min(image_data_pre))
                    / (np.max(image_data_pre) - np.min(image_data_pre))
                    * 255.0
                )
            else:
                lower_bound, upper_bound = np.percentile(
                    image_data[image_data > 0], 0.5
                ), np.percentile(image_data[image_data > 0], 99.5)
                image_data_pre = np.clip(image_data, lower_bound, upper_bound)
                image_data_pre = (
                    (image_data_pre - np.min(image_data_pre))
                    / (np.max(image_data_pre) - np.min(image_data_pre))
                    * 255.0
                )
                image_data_pre[image_data == 0] = 0

            image_data_pre = np.uint8(image_data_pre)
            img_roi = image_data_pre[z_index, :, :]
            # save the each slice as npy file
            for i in range(img_roi.shape[0]):
                img_i = img_roi[i, :, :]
                gt_i = gt_roi[i, :, :]
                img_3c = np.repeat(img_i[:, :, None], 3, axis=-1)
                resize_img_skimg = transform.resize(
                    img_3c,
                    (image_size, image_size),
                    order=3,
                    preserve_range=True,
                    mode="constant",
                    anti_aliasing=True,
                )
                resize_img_skimg_01 = (resize_img_skimg - resize_img_skimg.min()) / np.clip(
                    resize_img_skimg.max() - resize_img_skimg.min(), a_min=1e-8, a_max=None
                )  # normalize to [0, 1], (H, W, 3)
                resize_gt_skimg = transform.resize(
                    gt_i,
                    (image_size, image_size),
                    order=0,
                    preserve_range=True,
                    mode="constant",
                    anti_aliasing=False,
                )
                resize_gt_skimg = np.uint8(resize_gt_skimg)
                
                # Process lung mask
                lung_i = lung_roi[i, :, :]
                resize_lung_skimg = transform.resize(
                    lung_i,
                    (image_size, image_size),
                    order=0,
                    preserve_range=True,
                    mode="constant",
                    anti_aliasing=False,
                )
                resize_lung_skimg = np.uint8(resize_lung_skimg)
                
                np.save(
                    join(
                        npy_path,
                        "imgs",
                        base_name + "-" + str(i).zfill(3) + ".npy",
                    ),
                    resize_img_skimg_01,
                )
                np.save(
                    join(
                        npy_path,
                        "gts",
                        base_name + "-" + str(i).zfill(3) + ".npy",
                    ),
                    resize_gt_skimg,
                )
                np.save(
                    join(
                        npy_path,
                        "lungs",
                        base_name + "-" + str(i).zfill(3) + ".npy",
                    ),
                    resize_lung_skimg,
                )
