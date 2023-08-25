#!/bin/bash 
config_input=./data/images/PSR                          # Input directory
config_mask=./data/masks/tissue_background_masks/PSR    # Output directory
config_out=./data/data_PSR_configuration1.yaml           # Output directory

rm -r /home/user/source/pathology-common
cp -r ./pathology-common /home/user/source/pathology-common

python3 /home/user/source/pathology-common/scripts/createdataconfig.py \
        --images ${config_input} \
        --masks ${config_mask}/{image}_tb_mask.tif \
        --output ${config_out} \
        --purposes "{'training':0.75,'validation':0.25}" \
        --labels [1] \
        --level 1


config_input=./data/images/HE_L                         # Input directory
config_mask=./data/masks/tissue_background_masks/HE_L    # Output directory
config_out=./data/data_HE_L_configuration1.yaml           # Output directory

rm -r /home/user/source/pathology-common
cp -r ./pathology-common /home/user/source/pathology-common

python3 /home/user/source/pathology-common/scripts/createdataconfig.py \
        --images ${config_input} \
        --masks ${config_mask}/{image}_tb_mask.tif \
        --output ${config_out} \
        --purposes "{'training':0.75,'validation':0.25}" \
        --labels [1] \
        --level 1