#!/bin/bash

# Parameters
target=portal4
type=HE_L_2ndbatch
add_background=true
create_config=false
spacing=4.0
labels="{'steatosis': 1, 'inflammation': 2, 'liver':3, 'background':4, 'portal':5, 'portalField':6, 'liverInfBorder':7}"

# Paths
setup_script=./pathology-NASH-quantification/scripts/bash/setup_env.sh
image_dir=./data/images/${type}
annotation_dir=./data/annotations
base_mask_dir=./data/masks
jobName=portalFP
config_out_dir=./data/configFiles


if [ ! -f /home/user/source/deps.txt ]; then
    bash ${setup_script}
fi

class_mask_path=${base_mask_dir}/${jobName}/${target}_class_masks/${type}/${spacing}
mkdir -p ${class_mask_path}
python3 /home/user/source/pathology-common/scripts/convertannotations.py \
    --image ${image_dir} \
    --annotation ${annotation_dir}/${target} \
    --mask ${class_mask_path}/{image}_mask.tif \
    --labels "{'steatosis': 1, 'inflammation': 2, 'liver':3, 'background':4, 'portal':5, 'portalField':6, 'liverInfBorder':7}" \
    --spacing ${spacing} \

mask_path=${class_mask_path}
out_path=${config_out_dir}/data_${type}_${spacing}_${target}_configuration_${jobName}.yaml

if [ "$add_background" = true ]; then
    bg_mask_path=${base_mask_dir}/tissue_background_masks/${type}/images
    mask_path=${base_mask_dir}/${target}_combined/${type}/${spacing}
    if [ ! "$spacing" = "2.0" ]; then
        bg_annotation_dir=${annotation_dir}/background/${type}
        if [ ! -d ${bg_annotation_dir} ]; then
            mkdir -p ${bg_annotation_dir}
            python3 /home/user/source/pathology-common/scripts/convertmasks.py \
                --mask ${bg_mask_path} \
                --annotation ${bg_annotation_dir}/{mask}.xml \
                --conversion_spacing 2.0 \
                --target_spacing 0.25 \
                --grouping "{'background': 0, 'tissue':1}"
        fi
        bg_mask_path=${base_mask_dir}/rescaled_background_masks/${type}/${spacing}
        if [ ! -d ${base_mask_dir}/rescaled_background_masks/${type}/${spacing} ] ; then
            mkdir -p ${base_mask_dir}/rescaled_background_masks/${type}/${spacing}
            python3 /home/user/source/pathology-common/scripts/convertannotations.py \
                --image ${image_dir} \
                --annotation ${bg_annotation_dir}/{image}_tb_mask.xml \
                --mask ${bg_mask_path}/{image}_tb_mask.tif \
                --labels "{'background': 0, 'tissue':1}" \
                --spacing ${spacing} \
                --overwrite
        fi
        mask_path=${base_mask_dir}/${jobName}/${target}_combined/${type}/${spacing}_rescback
    fi
    mkdir -p ${mask_path}
    python3 /home/user/source/pathology-common/scripts/combinemasks.py \
            --left ${class_mask_path}/{base}_mask.tif \
            --right ${bg_mask_path}/{base}_tb_mask.tif \
            --result ${mask_path}/{base}_mask.tif \
            --operand + \
            --base '*' \
            --overwrite
    out_path=${config_out_dir}/data_${type}_${spacing}_${target}_configuration_${jobName}_bg.yaml
fi

if [ "$create_config" = true ]; then
    python3 /home/user/source/pathology-common/scripts/createdataconfig.py \
        --images "${image_dir}/*.tif" \
        --masks ${mask_path}/{image}_mask.tif \
        --output ${out_path} \
        --purposes "{'training':0.6,'validation':0.2,'testing':0.2}" \
        --labels [1,2,3,4,5,6,7,8] \
        --spacing ${spacing} \
        --overwrite
fi