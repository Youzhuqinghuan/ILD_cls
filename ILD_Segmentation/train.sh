#!/bin/bash

# 定义变量
IMG_PATH="/home/huchengpeng/ILD/imagesTr"
IMG_NAME_SUFFIX="_0000.nii.gz"
GT_PATH="/home/huchengpeng/ILD/labelsTr"
GT_NAME_SUFFIX=".nii.gz"
OUTPUT_PATH="./data/ILD"
NUM_WORKERS=4
MODALITY="CT"
ANATOMY="ILD"
WINDOW_LEVEL=40
WINDOW_WIDTH=400
SAVE_NII="--save_nii"

NPZ_TRAIN_DIR="./data/ILD/npz_train/CT_ILD"
NPZ_VALID_DIR="./data/ILD/npz_valid/CT_ILD"
NPY_DIR="./data/ILD/npy"

WORK_DIR="./work_dir"
EPOCHS=500
BATCH_SIZE=16
L_RATE=1e-5

# Tiny checkpoint
TASK_NAME="MedSAM2-Tiny-ILD"
PRETRAIN_MODEL_PATH="./checkpoints/sam2_hiera_tiny.pt"
MODEL_CFG="sam2_hiera_t.yaml"
MEDSAM2CHECKPOINT="./work_dir/MedSAM2-Tiny-ILD-20240911-1552/medsam_model_best.pth"

RESUME=false  # 将此值设置为true以继续训练
RESUME_PATH="./work_dir/MedSAM2-Tiny-ILD-20240822-2156/medsam_model_latest.pth"

TRAINORINFER=2 # Train: 1, Inference: 2 

USEMEDSAM=true

NPZ_TEST_DIR="./data/ILD/npz_test/CT_ILD"
PRED_SAVE_DIR="./segs/medsam2"
PRED_SAVE_DIR_SAM="./segs/sam2"
EVALPATH="./infer_eval/medsam2"
EVALPATH_SAM="./infer_eval/sam2"

VISUALIZE_PKL="./visualize/test_slice_indices.pkl"

if [ ! -e ${OUTPUT_PATH}/.ILD.npz.done ]; then
    python nii_to_npz.py \
        -img_path $IMG_PATH \
        -img_name_suffix $IMG_NAME_SUFFIX \
        -gt_path $GT_PATH \
        -gt_name_suffix $GT_NAME_SUFFIX \
        -output_path $OUTPUT_PATH \
        -num_workers $NUM_WORKERS \
        -modality $MODALITY \
        -anatomy $ANATOMY \
        -window_level $WINDOW_LEVEL \
        -window_width $WINDOW_WIDTH \
        -visualize $VISUALIZE_PKL \
        $SAVE_NII
    touch  ${OUTPUT_PATH}/.ILD.npz.done
fi

echo "Nii to Npz Completed!"

if [ ! -e ${OUTPUT_PATH}/.ILD.npy.done ]; then
    python npz_to_npy.py \
        -npz_train_dir $NPZ_TRAIN_DIR \
        -npz_valid_dir $NPZ_VALID_DIR \
        -npy_dir $NPY_DIR \
        -num_workers $NUM_WORKERS
    touch  ${OUTPUT_PATH}/.ILD.npy.done
fi

echo "Npz to Npy Completed!"

if [ "$TRAINORINFER" = 1 ]; then

    echo "Begin Training!"

    if [ "$RESUME" = true ]; then
        python finetune_sam2_img.py \
            -i $NPY_DIR \
            -task_name $TASK_NAME \
            -work_dir $WORK_DIR \
            -num_epochs $EPOCHS \
            -batch_size $BATCH_SIZE \
            -lr $L_RATE \
            -pretrain_model_path $PRETRAIN_MODEL_PATH \
            -model_cfg $MODEL_CFG \
            -resume $RESUME_PATH
    else
        python finetune_sam2_img.py \
            -i $NPY_DIR \
            -task_name $TASK_NAME \
            -work_dir $WORK_DIR \
            -num_epochs $EPOCHS \
            -batch_size $BATCH_SIZE \
            -pretrain_model_path $PRETRAIN_MODEL_PATH \
            -model_cfg $MODEL_CFG
    fi
else

    echo "Begin Inference!"

    if [ "$USEMEDSAM" = true ]; then
        python infer_medsam2_ILD.py \
            -data_root $NPZ_TEST_DIR \
            -pred_save_dir $PRED_SAVE_DIR \
            -sam2_checkpoint $PRETRAIN_MODEL_PATH \
            -medsam2_checkpoint $MEDSAM2CHECKPOINT \
            -model_cfg $MODEL_CFG \
            -bbox_shift 5 \
            -num_workers 10 \
            -vis $VISUALIZE_PKL \
            --visualize ## Save segmentation, ground truth volume, and images in .nii.gz for visualization

        python ./metrics/compute_metrics_ILD_all.py \
        -s ./${PRED_SAVE_DIR} \
        -g ${GT_PATH} \
        -csv_dir $EVALPATH
    else
        python infer_sam2_ILD.py \
            -data_root $NPZ_TEST_DIR \
            -pred_save_dir $PRED_SAVE_DIR_SAM \
            -sam2_checkpoint $PRETRAIN_MODEL_PATH \
            -model_cfg $MODEL_CFG \
            -bbox_shift 5 \
            -num_workers 10

        python ./metrics/compute_metrics_ILD_all.py \
        -s ./${PRED_SAVE_DIR_SAM} \
        -g ${GT_PATH} \
        -csv_dir $EVALPATH_SAM
    fi
fi