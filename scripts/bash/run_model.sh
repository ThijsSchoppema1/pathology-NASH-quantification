#!/bin/bash 

# Takes 20 min per epoch for 5k iterations, thus 20 hours
# Training params
config_file='./data/configFiles/train/train_simpleModel.yaml'

error_dir=./log_dir/runModel
out_path=./data/results/supervised/eb4_unet_s40_portal4_v2
error_dir=${out_path}

mkdir -p ${out_path}
mkdir -p ${error_dir}

if [ ! -f /home/user/source/deps.txt ]; then
  bash ./pathology-NASH-quantification/scripts/bash/setup_env.sh
fi

cp ${config_file} ${out_path}

python3 ./pathology-NASH-quantification/scripts/trainModel.py \
  --config_file=${config_file} > >(tee -a ${error_dir}/stdout.log) 2> >(tee -a ${error_dir}/stderr.log >&2)