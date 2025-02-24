#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
python pytorch/main.py \
    --name "mri_toep" \
    --epochs 24000 \
    --batch-size 1 \
    --lr 3e-5 \
    --lr-decay 0.99995 \
    --optim ams \
    --log-freq 1000 \
    --data-dir /home/jc_350/fastMRI/multicoil_train \
    --dataset mri_toep \
    --case-name file_brain_AXT1_201_6002779 \
    --slice-idx 3 \
    --mask-type toep_mask \
    --operator-type toeplitz \
    --mri-train-type inverse \
    model SL \
    --class-type toeplitz_symmetric \
    --r 8 \
    --is-complex \
    --dim 2
    