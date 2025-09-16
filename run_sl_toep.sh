#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

# Default train type
TYPE="inverse"

# Parse command line arguments
while [ "$#" -gt 0 ]; do
    case "$1" in
        --type)
            TYPE="$2"
            shift 2
            ;;
        *)
            break
            ;;
    esac
done

python pytorch/main.py \
    --name "mri_toep" \
    --epochs 24000 \
    --batch-size 1 \
    --lr 3e-5 \
    --lr-decay 0.99995 \
    --optim ams \
    --log-freq 100 \
    --data-dir /home/jc_350/fastMRI/multicoil_train \
    --dataset mri_toep \
    --case-name file_brain_AXT1_201_6002779 \
    --slice-idx 3 \
    --mask-type toep_mask \
    --operator-type toeplitz \
    --mri-train-type $TYPE \
    model SL \
    --class-type toeplitz_symmetric \
    --r 6 \
    --is-complex \
    --dim 2
    