#!/bin/bash 

# Takes 20 min per epoch for 5k iterations, thus 20 hours
# Training params
source_file='./data/configFiles/data_HE_L_0.5_precise_configuration.yaml'
out_path=./data/results/supervised/unet/efficientnet_s05
error_dir=./log_dir/sup

patch_size=224
iterations=1000
batch_size=16
n_classes=3
level=1
epochs=100
dropout=0.3
lr=0.0005
agb=1
model=unet
spacing=0.5

# Testing params
test_model=false
in_img_dir=./data/images/raw_images
out_seg_dir=./data/results/unet/efficientnet/segmentations
model_path=${out_path}/last.ckpt
error_dir=./log_dir
color_type="HE_L"

mkdir -p ${out_path}
mkdir -p ${error_dir}

if [ ! -f /home/user/source/deps.txt ]; then
  bash ./pathology-NASH-quantification/scripts/bash/setup_env.sh
fi

python3 ./pathology-NASH-quantification/scripts/trainSupervisedModel.py \
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
  --level=${level} > >(tee -a ${error_dir}/stdout.log) 2> >(tee -a ${error_dir}/stderr.log >&2)

if [ "$test_model" = true ] ; then
  python3 ./pathology-NASH-quantification/scripts/ApplySupervisedModel.py \
    --in_dir=${in_img_dir} \
    --out_dir=${out_seg_dir} \
    --color_type=${color_type} \
    --modelpath=${model_path} \
    --level=${level} > >(tee -a ${error_dir}/stdout.log) 2> >(tee -a ${error_dir}/stderr.log >&2)
fi
