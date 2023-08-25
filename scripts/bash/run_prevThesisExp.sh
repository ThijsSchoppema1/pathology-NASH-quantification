#!/bin/bash 

in_dir=./data/images/raw_images
out_dir=./data/results/previousThesis3
error_dir=./log_dir/prevT
color_type="HE_L"
level=1
partial=true
window_size=8192

mkdir -p ${out_dir}/masks_${color_type}
mkdir -p ${error_dir}

if [ ! -f /home/user/source/deps.txt ]; then
  bash ./pathology-NASH-quantification/scripts/bash/setup_env.sh
fi

python3 ./pathology-NASH-quantification/scripts/runPreviousThesis.py \
  --in_dir=${in_dir} \
  --out_dir=${out_dir}/masks_${color_type} \
  --color_type=${color_type} \
  --partial=${partial} \
  --window_size=${window_size} \
  --level=${level} > >(tee -a ${error_dir}/stdout.log) 2> >(tee -a ${error_dir}/stderr.log >&2)

# color_type="PSR"
# level=3
# partial=false

# mkdir -p ${out_dir}/masks_${color_type}
# mkdir -p ${error_dir}

# python3 ./pathology-NASH-quantification/scripts/runPreviousThesis.py \
#   --in_dir=${in_dir} \
#   --out_dir=${out_dir}/masks_${color_type} \
#   --color_type=${color_type} \
#   --partial=${partial} \
#   --window_size=${window_size} \
#   --level=${level} > >(tee -a ${error_dir}/stdout.log) 2> >(tee -a ${error_dir}/stderr.log >&2)