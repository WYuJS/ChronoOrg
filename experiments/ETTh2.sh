#!/bin/bash

# pred_len=96
python3 train.py \
  --model_name chronoorg \
  --data_type etth2 \
  --seq_len 96 \
  --pred_len 96 \
  --patch_len 24 \
  --patch_stride 12 \
  --d_model 128 \
  --dropout 0.15 \
  --pooling last \
  --stage_dims 128 192 256 \
  --depths 2 2 2 \
  --num_heads 8 4 4  \
  --window_size 4 4 4 \
  --affine 1 \
  --mlp_ratio 2


# pred_len=192
python3 train.py \
  --model_name chronoorg \
  --data_type etth2 \
  --seq_len 96 \
  --pred_len 192 \
  --patch_len 16 \
  --patch_stride 8 \
  --d_model 128 \
  --dropout 0.3 \
  --pooling last \
  --stage_dims 128 192 256 \
  --depths 2 2 2 \
  --num_heads 2 2 2 \
  --window_size 4 4 4 \
  --affine 1 \
  --mlp_ratio 2

# pred_len=336
python3 train.py \
  --model_name chronoorg \
  --data_type etth2 \
  --seq_len 96 \
  --pred_len 336 \
  --patch_len 24 \
  --patch_stride 12 \
  --d_model 128 \
  --dropout 0.15 \
  --pooling last \
  --stage_dims 192 192 256 \
  --depths 2 2 2 \
  --num_heads 8 4 4 \
  --window_size 12 10 8 \
  --affine 1 \
  --mlp_ratio 2


# pred_len=720
python3 train.py \
  --model_name chronoorg \
  --data_type etth2 \
  --seq_len 96 \
  --pred_len 720 \
  --patch_len 24 \
  --patch_stride 12 \
  --d_model 128 \
  --dropout 0.5 \
  --pooling last \
  --stage_dims 168 192 256 \
  --depths 2 2 2 \
  --num_heads 4 2 2 \
  --window_size 4 4 4 \
  --affine 0 \
  --mlp_ratio 2

