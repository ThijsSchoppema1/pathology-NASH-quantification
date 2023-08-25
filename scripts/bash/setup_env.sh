#!/bin/bash 

echo "Installing deps"

/usr/local/bin/python3.8 -m pip install --upgrade pip
pip3 install -r ./pathology-NASH-quantification/requirements.txt
pip3 install -U typing_extensions

rm -r /home/user/source/pathology-common
rm -r /home/user/source/pathology-fast-inference
cp -r ./pathology-common /home/user/source/pathology-common
cp -r ./pathology-fast-inference /home/user/source/pathology-fast-inference

cp ./pathology-NASH-quantification/scripts/customProcessors/*.py /home/user/source/pathology-fast-inference/fastinference/processors/

touch /home/user/source/deps.txt