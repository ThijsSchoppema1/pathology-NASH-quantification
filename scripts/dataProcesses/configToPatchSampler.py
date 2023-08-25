# Create patch samples from the config for a MIL trainer

import yaml
from pathlib import Path

import digitalpathology.generator.patch.patchsampler as psampler
import digitalpathology.generator.patch.patchsource as psource

def parse_config(config_file, partition):
    with open(config_file, 'r') as stream:
        data_loaded = yaml.safe_load(stream)

    root_path = data_loaded['path']['root']
    base_image_path = data_loaded['path']['images']
    base_mask_path = data_loaded['path']['masks']
    
    data_set = data_loaded['data'][partition]['default']

    psource_list = []
    path_scores = []
    for sample in data_set:
        image_path = sample['image'].format(root=root_path, images=base_image_path)
        mask_path = sample['mask'].format(root=root_path, masks=base_mask_path)
        labels = tuple(sample['labels'])

        if 'path_scores' in sample:
            path_scores.append(sample['path_scores'])

        psource_list.append(
            psource.PatchSource(image_path, mask_path, None, labels)
        )

    return psource_list, path_scores

def create_samplers(source_list, mask_spacing, spacing_tol, in_channels, label_mode, cache_path):
    sampler_list = []
    for source_item in source_list:
        sampler_list.append(
            psampler.PatchSampler(
                patch_source=source_item,
                create_stat=False,
                mask_spacing=mask_spacing,
                spacing_tolerance=spacing_tol,
                input_channels=in_channels,
                label_mode=label_mode,
                cache_path=cache_path
            )
        )

    return sampler_list
