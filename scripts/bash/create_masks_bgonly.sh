#!/bin/bash

# Parameters
target=direct_PSR
type=PSR
add_background=true
create_config=true
spacing=0.5

# Paths
setup_script=./pathology-NASH-quantification/scripts/bash/setup_env.sh
image_dir=./data/images/${type}_2ndbatch
annotation_dir=./data/annotations
base_mask_dir=./data/masks
config_out_dir=./data/configFiles


if [ ! -f /home/user/source/deps.txt ]; then
    bash ${setup_script}
fi

out_path=${config_out_dir}/data_${type}_${spacing}_${target}_configuration.yaml
bg_mask_path=${base_mask_dir}/tissue_background_masks/${type}/images
if [ ! "$spacing" = "2.0" ]; then
    bg_annotation_dir=${annotation_dir}/background/${type}
    if [ ! -d ${bg_annotation_dir} ]; then
        mkdir -p ${bg_annotation_dir}
    fi
    python3 /home/user/source/pathology-common/scripts/convertmasks.py \
        --mask ${bg_mask_path} \
        --annotation ${bg_annotation_dir}/{mask}.xml \
        --conversion_spacing 2.0 \
        --target_spacing 0.25 \
        --grouping "{'background': 0, 'tissue':1}"
    # fi
    bg_mask_path=${base_mask_dir}/rescaled_background_masks/${type}/${spacing}
    if [ ! -d ${base_mask_dir}/rescaled_background_masks/${type}/${spacing} ] ; then
        mkdir -p ${base_mask_dir}/rescaled_background_masks/${type}/${spacing}
    fi
    python3 /home/user/source/pathology-common/scripts/convertannotations.py \
        --image ${image_dir} \
        --annotation ${bg_annotation_dir}/{image}_tb_mask.xml \
        --mask ${bg_mask_path}/{image}_tb_mask.tif \
        --labels "{'background': 0, 'tissue':1}" \
        --spacing ${spacing}
    # fi
fi

if [ "$create_config" = true ]; then
    python3 /home/user/source/pathology-common/scripts/createdataconfig.py \
        --images "${image_dir}/*.tif" \
        --masks ${bg_mask_path}/{image}_tb_mask.tif \
        --output ${out_path} \
        --purposes "{'training':0.6,'validation':0.2,'testing':0.2}" \
        --labels [0,1] \
        --spacing ${spacing} \
        --overwrite
fi