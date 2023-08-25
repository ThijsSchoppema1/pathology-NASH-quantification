#!/bin/bash

target=precise
type=HE_L
add_background=false
create_config=true
spacing=2.0

if [ ! -f /home/user/source/deps.txt ]; then
    bash ./pathology-NASH-quantification/scripts/bash/setup_env.sh
fi

mkdir -p ./data/masks/${target}_class_masks/${type}/${spacing}
python3 /home/user/source/pathology-common/scripts/convertannotations.py \
    --image ./data/images/${type} \
    --annotation ./data/annotations/${target} \
    --mask ./data/masks/${target}_class_masks/${type}/${spacing}/{image}_mask.tif \
    --labels "{'steatosis': 1, 'inflammation':2}" \
    --spacing ${spacing} \
    # --overwrite

mask_path=./data/masks/${target}_class_masks/${type}/${spacing}/{image}_mask.tif
if [ "$add_background" = true ]; then
    mkdir -p ./data/masks/${target}_combined/${type}/
    python3 /home/user/source/pathology-common/scripts/combinemasks.py \
            --left ./data/masks/${target}_class_masks/${type}/${spacing}/{base}_mask.tif \
            --right ./data/masks/tissue_background_masks/${type}/{base}_tb_mask.tif \
            --result ./data/masks/${target}_combined/${type}/${spacing}/{base}_mask.tif \
            --operand + \
            --base '*' \
            --overwrite
    mask_path=./data/masks/${target}_combined/${type}/${spacing}/{image}_mask.tif
fi


if [ "$create_config" = true ]; then
    python3 /home/user/source/pathology-common/scripts/createdataconfig.py \
        --images './data/images/HE_L/*.tif' \
        --masks ${mask_path} \
        --output ./data/configFiles/data_${type}_${spacing}_${target}_nobg_configuration.yaml \
        --purposes "{'training':0.7,'validation':0.2, 'testing':0.1}" \
        --labels [0,1,2] \
        --spacing ${spacing} \
        --overwrite
fi