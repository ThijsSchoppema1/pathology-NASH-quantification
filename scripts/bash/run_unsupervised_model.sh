#!/bin/bash 

# Takes 20 min per epoch for 5k iterations, thus 20 hours
# Training params
source_file='./data/configFiles/data_HE_L_bgres_configuration.yaml'

out_path=./data/results/wnet/classes64lvl1sp05_r3
resume_checkpoint=./data/results/wnet/classes64lvl1sp05_r2


error_dir=./log_dir/unsup

patch_size=224
iterations=1000
batch_size=10
n_classes=64
level=1
spacing=0.5
epochs=150
dropout=0.3
lr=0.0003
agb=1
model=wnet2

mkdir -p ${out_path}
mkdir -p ${error_dir}

if [ ! -f /home/user/source/deps.txt ]; then
  bash ./pathology-NASH-quantification/scripts/bash/setup_env.sh
fi

python3 ./pathology-NASH-quantification/scripts/trainUnsupervisedModel.py \
  --source_file=${source_file} \
  --patch_size=${patch_size} \
  --iterations=${iterations} \
  --batch_size=${batch_size} \
  --n_classes=${n_classes} \
  --epochs=${epochs} \
  --out_path=${out_path} \
  --dropout=${dropout} \
  --lr=${lr} \
  --accumulate_grad_batches=${agb} \
  --model_type=${model} \
  --spacing=${spacing} \
  --resume_checkpoint=${resume_checkpoint} \
  --level=${level} > >(tee -a ${error_dir}/stdout.log) 2> >(tee -a ${error_dir}/stderr.log >&2)