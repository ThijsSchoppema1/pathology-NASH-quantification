# pathology-NASH-quantification

## Required packages & repositiories
* Pathology-common: rework branch, Radboud UMC DIAG
* Pathology-fast-inference: master branch, Radboud UMC DIAG
* Requirements.txt
** Requirements_version_all contains package versions of each package in the environment
** Requirements_version contains package versions of the main packages

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
