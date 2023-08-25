#!/bin/bash 

# Takes 20 min per epoch for 5k iterations, thus 20 hours
# Training params
config_file='./data/configFiles/train/train_ltsFocused.yaml'
config_file2='./data/configFiles/train/train_ltsFocusedContinued.yaml'

error_dir=./log_dir/runModel

out_path=./data/results/lts_results/lts_eb2_unet
out_path2=./data/results/lts_results/lts_eb2_unet_continued
error_dir=${out_path}

mkdir -p ${out_path}
mkdir -p ${error_dir}

cp ${config_file} ${out_path}

if [ ! -f /home/user/source/deps.txt ]; then
  bash ./pathology-NASH-quantification/scripts/bash/setup_env.sh
fi

python3 ./pathology-NASH-quantification/scripts/trainModel.py \
  --config_file=${config_file} > >(tee -a ${error_dir}/stdout.log) 2> >(tee -a ${error_dir}/stderr.log >&2)

# Continue training
error_dir=${out_path2}

mkdir -p ${out_path2}
mkdir -p ${error_dir}

cp ${config_file2} ${out_path2}

python3 ./pathology-NASH-quantification/scripts/trainModel.py \
  --config_file=${config_file2} > >(tee -a ${error_dir}/stdout.log) 2> >(tee -a ${error_dir}/stderr.log >&2)


