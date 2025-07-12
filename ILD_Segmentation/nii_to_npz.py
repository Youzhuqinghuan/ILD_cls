# -*- coding: utf-8 -*-
# %% import packages
# pip install connected-components-3d
import numpy as np

# import nibabel as nib
import SimpleITK as sitk
import os

join = os.path.join
from tqdm import tqdm
import cc3d

import multiprocessing as mp
from functools import partial

import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-modality", type=str, default="CT", help="CT or MR, [default: CT]")
parser.add_argument("-anatomy", type=str, default="Abd",
                    help="Anaotmy name, [default: Abd]")
parser.add_argument("-img_name_suffix", type=str, default="_0000.nii.gz",
                    help="Suffix of the image name, [default: .nii.gz]")
parser.add_argument("-gt_name_suffix", type=str, default=".nii.gz",
                    help="Suffix of the ground truth name, [default: .nii.gz]")
parser.add_argument("-img_path", type=str, default="data/FLARE22Train/images",
                    help="Path to the nii images, [default: data/FLARE22Train/images]")
parser.add_argument("-gt_path", type=str, default="data/FLARE22Train/labels",
                    help="Path to the ground truth, [default: data/FLARE22Train/labels]")
parser.add_argument("-output_path", type=str, default="data/npz/Flare22Train",
                    help="Path to save the npy files, [default: ./data/npz]")
parser.add_argument("-num_workers", type=int, default=4,
                    help="Number of workers, [default: 4]")
parser.add_argument("-window_level", type=int, default=40,
                    help="CT window level, [default: 40]")
parser.add_argument("-window_width", type=int, default=400,
                    help="CT window width, [default: 400]")
parser.add_argument("--save_nii", action="store_true",
                    help="Save the image and ground truth as nii files for sanity check; they can be removed")
parser.add_argument("-visualize", type=str, default="./visualize/test_slice_indices.pkl")

args = parser.parse_args()

# convert nii image to npz files, including original image and corresponding masks
modality = args.modality  # CT or MR
anatomy = args.anatomy  # anantomy + dataset name
img_name_suffix = args.img_name_suffix  # "_0000.nii.gz"
gt_name_suffix = args.gt_name_suffix  # ".nii.gz"
prefix = modality + "_" + anatomy + "_"

nii_path = args.img_path  # path to the nii images
gt_path = args.gt_path  # path to the ground truth
output_path = args.output_path  # path to save the preprocessed files
npz_tr_path = join(output_path, "npz_train", prefix[:-1])
os.makedirs(npz_tr_path, exist_ok=True)
npz_va_path = join(output_path, "npz_valid", prefix[:-1])
os.makedirs(npz_va_path, exist_ok=True)
npz_ts_path = join(output_path, "npz_test", prefix[:-1])
os.makedirs(npz_ts_path, exist_ok=True)

num_workers = args.num_workers

names = sorted(os.listdir(gt_path))
print(f"ori \# files {len(names)=}")
names = [
    name
    for name in names
    if os.path.exists(join(nii_path, name.split(gt_name_suffix)[0] + img_name_suffix))
]
print(f"after sanity check \# files {len(names)=}")

# set window level and width
# https://radiopaedia.org/articles/windowing-ct
WINDOW_LEVEL = args.window_level # only for CT images
WINDOW_WIDTH = args.window_width # only for CT images

save_nii = args.save_nii
# %% save preprocessed images and masks as npz files
def preprocess(name, npz_path):
    """
    Preprocess the image and ground truth, and save them as npz files

    Parameters
    ----------
    name : str
        name of the ground truth file
    npz_path : str
        path to save the npz files
    """
    image_name = name.split(gt_name_suffix)[0] + img_name_suffix
    gt_name = name
    gt_sitk = sitk.ReadImage(join(gt_path, gt_name))
    gt_data_ori = np.uint8(sitk.GetArrayFromImage(gt_sitk))

    voxel_num_thre2d = 100
    voxel_num_thre3d = 1000
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
        # crop the ground truth with non-zero slices
        gt_roi = gt_data_ori[z_index, :, :]
        # load image and preprocess
        img_sitk = sitk.ReadImage(join(nii_path, image_name))
        image_data = sitk.GetArrayFromImage(img_sitk)
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
        np.savez_compressed(join(npz_path, gt_name.split(gt_name_suffix)[0]+'.npz'), imgs=img_roi, gts=gt_roi, spacing=img_sitk.GetSpacing())

        # save the image and ground truth as nii files for sanity check;
        # they can be removed
        if save_nii:
            img_roi_sitk = sitk.GetImageFromArray(img_roi)
            img_roi_sitk.SetSpacing(img_sitk.GetSpacing())
            sitk.WriteImage(
                img_roi_sitk,
                join(npz_path, gt_name.split(gt_name_suffix)[0] + "_img.nii.gz"),
            )
            gt_roi_sitk = sitk.GetImageFromArray(gt_roi)
            gt_roi_sitk.SetSpacing(img_sitk.GetSpacing())
            sitk.WriteImage(
                gt_roi_sitk,
                join(npz_path, gt_name.split(gt_name_suffix)[0] + "_gt.nii.gz"),
            )

def record_test_slice_indices(names, output_file):
    """
    Record the slice indices and dimensions for test set images and save to a pickle file.

    Parameters
    ----------
    names : list
        List of ground truth filenames.
    output_file : str
        Path to save the pickle file.
    """
    test_slice_indices = {}

    for name in names:
        gt_name = name
        gt_sitk = sitk.ReadImage(os.path.join(gt_path, gt_name))
        gt_data_ori = np.uint8(sitk.GetArrayFromImage(gt_sitk))

        # Record the dimensions of the image
        dimensions = gt_data_ori.shape

        voxel_num_thre2d = 100
        voxel_num_thre3d = 1000

        # Remove small objects in 3D
        gt_data_ori = cc3d.dust(
            gt_data_ori, threshold=voxel_num_thre3d, connectivity=26, in_place=True
        )
        
        # Remove small objects in 2D
        for slice_i in range(gt_data_ori.shape[0]):
            gt_i = gt_data_ori[slice_i, :, :]
            gt_data_ori[slice_i, :, :] = cc3d.dust(
                gt_i, threshold=voxel_num_thre2d, connectivity=8, in_place=True
            )

        # Find non-zero slices
        z_index, _, _ = np.where(gt_data_ori > 0)
        z_index = np.unique(z_index)

        if len(z_index) > 0:
            # Store both slice indices and dimensions
            test_slice_indices[name] = {'slice_indices': z_index.tolist(), 'dimensions': dimensions}

    # Save to pickle file
    with open(output_file, 'wb') as f:
        pickle.dump(test_slice_indices, f)

if __name__ == "__main__":
    total_names = len(names)
    train_size = int(total_names * 0.7)
    val_size = int(total_names * 0.1)
    test_size = total_names - train_size - val_size

    # tr_names = names[:train_size]
    # va_names = names[train_size:train_size + val_size]
    # ts_names = names[train_size + val_size:]
    
    # Hard code
    va_names = ['CT_UIP21.nii.gz', 'CT_UIP8.nii.gz', 'CT_NSIP11.nii.gz']
    ts_names = ['CT_NSIP12.nii.gz', 'CT_NSIP10.nii.gz', 'CT_NSIP16.nii.gz', 'CT_UIP8.nii.gz', 'CT_UIP4.nii.gz', 'CT_NSIP18.nii.gz', 'CT_UIP5.nii.gz', 'CT_UIP1.nii.gz', 'CT_UIP18.nii.gz']
    tr_names = [
        name
        for name in names
        if name not in va_names and name not in ts_names
                ]

    preprocess_tr = partial(preprocess, npz_path=npz_tr_path)
    preprocess_va = partial(preprocess, npz_path=npz_va_path)
    preprocess_ts = partial(preprocess, npz_path=npz_ts_path)

    with mp.Pool(num_workers) as p:
        with tqdm(total=len(tr_names)) as pbar:
            pbar.set_description("Preprocessing training data")
            for i, _ in tqdm(enumerate(p.imap_unordered(preprocess_tr, tr_names))):
                pbar.update()
        with tqdm(total=len(va_names)) as pbar:
            pbar.set_description("Preprocessing validation data")
            for i, _ in tqdm(enumerate(p.imap_unordered(preprocess_va, va_names))):
                pbar.update()
        with tqdm(total=len(ts_names)) as pbar:
            pbar.set_description("Preprocessing testing data")
            for i, _ in tqdm(enumerate(p.imap_unordered(preprocess_ts, ts_names))):
                pbar.update()

    output_file = args.visualize
    record_test_slice_indices(ts_names, output_file)
    print(f"Test slice indices have been recorded in {output_file}.")