# pathology-NASH-quantification

## Required packages & repositiories
* Pathology-common: rework branch, commit: 33d23ee
* Pathology-fast-inference: master branch, commit: de73e99
* Requirements.txt
** Requirements_version_all contains package versions of each package in the environment
** Requirements_version contains package versions of the main packages

## Final model weights
The final model weights of both segmentation models can be found at:
`./final_weights`
With `segment_portal` containing the weights of the portal model and `segment_steatosis_inflammation` of the steatosis & inflammation model

## Docker
The segmentation models & post-processing pipeline can be run with the docker `doduo1.umcn.nl/thijs_schoppema/NASH_quantification:1.0`.
The docker has as entrypoint `run.sh` found at `./docker/run.sh`.
This file requires 3 different inputs:
* The folder containing the HE&E WSIs with spacings of 0.5 and 4.0
* A folder containing tissue background masks created with the docker `doduo1.umcn.nl/peter/algorithm-tissue-background-segmentation:2`. The result should be a tissue-background mask with a spcacing of 2.0
* The output folder to store the results

## Project page
Further information about the project can be found on:
* https://grand-challenge.org/algorithms/automated-nash-grading-of-liver-histopathology/
* https://www.ai-for-health.nl/projects/ai4h_msc_nash/

## Segmentation pipeline
* The background masks were created with the script: `jobScripts/schedule_tissue_background.sh`
* The tissue masks & rescaled background masks were created with `scripts/bash/create_masks_resisedbg.sh`
* * For specifically rescaling the background mask, `scripts/bash/create_masks_bgonly.sh` is also possible
* In order to train the models the following configs and scripts have been used:
* * `scripts/bash/run_model.sh`
* * 0.5 steatosis/inflammation model: `./final_weights/segment_steatosis_inflammation/train_simpleModel.yaml`
* * 4.0 portal model: `./final_weights/segment_portal/train_simpleModel.yaml`
* Inference has been run with `scripts/bash/run_fast_inference2.sh`
* Post-processing and after testing has been done with `scripts/bash/combine_masks.sh`

## CLAM approach
* For the CLAM models and results the config files found in `./data/configFiles/clam/kfold` were used