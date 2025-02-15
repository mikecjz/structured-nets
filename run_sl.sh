#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
python pytorch/main.py \
    --name "mri_grappa" \
    --epochs 8000 \
    --batch-size 1 \
    --lr 2e-5 \
    --lr-decay 1 \
    --optim adam \
    --log-freq 200 \
    --data-dir /home/jc_350/fastMRI/multicoil_train \
    --dataset mri_grappa \
    --case-name file_brain_AXT1_201_6002779 \
    --slice-idx 3 \
    --mask-type toep_mask \
    --operator-type toeplitz \
    --mri-train-type inverse \
    model SL \
    --class-type toeplitz_symmetric \
    --r 6 \
    --is-complex \
    --dim 2
    