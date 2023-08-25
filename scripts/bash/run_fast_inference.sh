#!/bin/bash 

INPUTS="./data/images/HE_L/*.tif"
# INPUTSPATH="./data/images/HE_L/"
# MASK_DIR=./masks/results/precise_combined/HE_L/2.0/
# OUT=./data/results/supervised/eb4_unet_s05_bgfoc/wsi_preds
# MODEL_DIR=./data/results/supervised/eb4_unet_s05_bgfoc/model_best_statedict.pt
MODEL_DIR='HNE_1_2' #./data/results/wnet/classes64lvl1sp05/model_enc_statedict.pt
OUT=./data/results/previousThesis/HE_L_preds
PROCESSOR=simple_processor # PROCESSOR=torch_processor

ADDCUSTOM=true
TILESIZE=8192
BATCHSIZE=1
BACKGROUND=true

mkdir -p ${OUT}

if [ ! -f /home/user/source/deps.txt ]; then
  bash ./pathology-NASH-quantification/scripts/bash/setup_env.sh
fi

if [ "$ADDCUSTOM" = true ]; then
    cp ./pathology-NASH-quantification/scripts/customProcessors/*.py /home/user/source/pathology-fast-inference/fastinference/processors/
fi

python3 /home/user/source/pathology-fast-inference/scripts/applynetwork_multiproc.py \
    --input_wsi_path="$INPUTS" --output_wsi_path="$OUT/{image}_pred.tif" \
    --model_path="${MODEL_DIR}" \
    --read_spacing=0.5 --write_spacing=0.5 --mask_spacing=0.5 --tile_size=${TILESIZE} --batch_size=${BATCHSIZE} \
    --gpu_count=1 --readers=3 --writers=3 \
    --custom_processor ${PROCESSOR} --axes_order='cwh'

if [ "$BACKGROUND" = true ]; then
  bg_mask_path=./data/masks/rescaled_background_masks/HE_L/0.5/
  python3 /home/user/source/pathology-common/scripts/combinemasks.py \
            --left ${OUT}/{base}_pred.tif \
            --right ${bg_mask_path}/{base}_tb_mask.tif \
            --result ${OUT}/corrected/{base}_pred.tif \
            --operand '*' \
            --base '*' \
            --overwrite
fi