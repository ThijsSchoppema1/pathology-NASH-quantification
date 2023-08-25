#!/bin/bash

out_space=0.5
BACKGROUND=true

mask_dir1=./data/results/supervised/eb4_unet_s40_portal4_v2_last/wsi_preds
mask_dir2=./data/results/supervised/eb4_unet_s05_portal4_v4_last/wsi_preds
out_dir=./data/results/supervised/combined_portal4_s05v4_s40v2_last
image_dir=./data/images/${type}
bg_mask_path=./data/masks/rescaled_background_masks/HE_L/0.5/
# bg_mask_path=./data/masks/tissue_background_masks/HE_L/images
type=HE_L

if [ ! -f /home/user/source/deps.txt ]; then
  bash ./pathology-NASH-quantification/scripts/bash/setup_env.sh
fi

mkdir -p ${out_dir}/resmask
python3 /home/user/source/pathology-common/scripts/zoomimage.py \
    --image ${mask_dir1} \
    --template ${mask_dir2}/{image}.tif \
    --output ${out_dir}/resmask/{image}.tif \
    --spacing ${out_space} \
    --tolerance 3.5 

mkdir -p ${out_dir}/combined
python3 ./pathology-NASH-quantification/scripts/combineResults.py \
    --up_mask_dir ${out_dir}/resmask \
    --mask_dir ${mask_dir2} \
    --out_dir ${out_dir}/combined \
    --patched 

mkdir -p ${out_dir}/corrected
python3 /home/user/source/pathology-common/scripts/combinemasks.py \
    --left ${out_dir}/combined/{base}_pred.tif \
    --right ${bg_mask_path}/{base}_tb_mask.tif \
    --result ${out_dir}/corrected/{base}_pred.tif \
    --operand '*' \
    --base '*' 

mkdir -p ${out_dir}/post_processed
python3 ./pathology-NASH-quantification/scripts/postProcess.py \
  --in_dir ${out_dir}/corrected \
  --out_dir ${out_dir}/post_processed \
  --spacing 0.5

python3 ./pathology-NASH-quantification/scripts/inferResults.py \
  --in_dir ${out_dir}/post_processed

CONFIG='./data/configFiles/train/train_simpleModel_post.yaml'
python3 ./pathology-NASH-quantification/scripts/testModel.py \
  --config_file ${CONFIG} \
  --post_model true

python3 ./pathology-NASH-quantification/scripts/testModel.py \
  --config_file ${CONFIG} \
  --post_model true \
  --val_set true