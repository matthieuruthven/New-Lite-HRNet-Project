#!/bin/bash

# Lightbox (Lite-HRNet-18, augmentation, no pretraining)
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
       configs/satellite_2d_keypoint/topdown_heatmap/speedplus/td-hm_litehrnet-18_speedplus-640x640_lbx_to_lbx.py

# Lightbox (Lite-HRNet-30, augmentation, no pretraining)
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
       configs/satellite_2d_keypoint/topdown_heatmap/speedplus/td-hm_litehrnet-30_speedplus-640x640_lbx_to_lbx.py