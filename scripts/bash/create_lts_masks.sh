add_background=false
create_config=true
spacing=0.5

if [ ! -f /home/user/source/deps.txt ]; then
    bash ./pathology-NASH-quantification/scripts/bash/setup_env.sh
fi

mkdir -p ./data/LTSdata/annotationMask_${spacing}
python3 /home/user/source/pathology-common/scripts/convertannotations.py \
    --image ./data/LTSdata/WSI \
    --annotation ./data/LTSdata/annotationXML \
    --mask ./data/LTSdata/annotationMask_${spacing}/{image}_mask.tif \
    --labels "{'Steatosis': 1, 'Lymphocytes': 2, 'Necrosis and debris': 3}" \
    --spacing ${spacing}

    # Don't use Liver, else fix tif files

mask_path=./data/LTSdata/annotationMask_${spacing}/{image}_mask.tif
if [ "$add_background" = true ]; then
    mkdir -p ./data/LTSdata/combinedMask/
    python3 /home/user/source/pathology-common/scripts/combinemasks.py \
            --left ./data/LTSdata/annotationMask/{base}_mask.tif \
            --right ./data/LTSdata/tissue_background_masks/{base}_tb_mask.tif \
            --result ./data/LTSdata/combinedMask/{base}_mask.tif \
            --operand + \
            --base '*' \
            --overwrite
    mask_path=./data/LTSdata/combinedMask/{image}_mask.tif
fi

if [ "$create_config" = true ]; then
    python3 /home/user/source/pathology-common/scripts/createdataconfig.py \
        --images './data/LTSdata/WSI/*.mrxs' \
        --masks ${mask_path} \
        --output ./data/configFiles/data_LTS_${spacing}_configuration.yaml \
        --purposes "{'training':0.7,'validation':0.2, 'testing':0.1}" \
        --labels [1,2,3] \
        --spacing ${spacing} \
        --overwrite
fi