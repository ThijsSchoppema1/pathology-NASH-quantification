#!/bin/bash 

in_dir=./data/images/raw_images
out_dir=./data/results/wnet/untrainedModel
model_path=./data/results/wnet/classes64lvl1v3/model_last_statedict.pt
error_dir=./log_dir/unsup
color_type="HE_L"
level=1

mkdir -p ${out_dir}/segmentations
mkdir -p ${error_dir}

if [ ! -f /home/user/source/deps.txt ]; then
  bash ./pathology-NASH-quantification/scripts/bash/setup_env.sh
fi

python3 ./pathology-NASH-quantification/scripts/ApplyUnsupervisedModel.py \
  --in_dir=${in_dir} \
  --out_dir=${out_dir}/segmentations \
  --color_type=${color_type} \
  --modelpath=${model_path} \
  --level=${level} > >(tee -a ${error_dir}/stdout.log) 2> >(tee -a ${error_dir}/stderr.log >&2)