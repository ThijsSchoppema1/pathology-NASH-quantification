# Data generators used by train_model and test_model
# UnSupDataset, is most often used, standard data extraction with mask
# WSIInferenceSet
# WSIMILset, extract patches from a WSI and create a WSI level label
# imagePatchSampler, extract patches from a folder and create a WSI level label
# imagePatchSamplerInference

from dataProcesses import augmentations
from functools import partial
from torch.utils.data import Dataset
import multiresolutionimageinterface as mir
import numpy as np

import dataProcesses.configToPatchSampler as CPS
import torch.nn.functional as F
import torch
import yaml

from pathlib import Path
from PIL import Image

class UnSupDataset(Dataset):
    def __init__(self, data_gen, augmentations_pipeline=None, toFloat=False):
        self._data_gen = data_gen
        self._aug_fn = augmentations_pipeline
        self.totensor = partial(augmentations.vaugment_fn, transform = augmentations.val_transforms)
        self.toFloat = toFloat

    def __len__(self):
        return self._data_gen._iterations

    def __getitem__(self, idx):
        batch, masks = self._preprocess_batch(idx)
        return batch, masks
    
    def _preprocess_batch(self, idx):
        patch, mask, weights = self._data_gen[idx]
        if self._aug_fn:
            patch, mask = self._aug_fn(patch, mask)
        else:
            patch, mask = self.totensor(patch, mask)
        if self.toFloat:
            return patch, mask

        return patch / 255.0, mask.float()

class WSIInferenceSet(Dataset):
    def __init__(self, img_file, level, patch_size, device, totensor=True, mask_file=None):
        self.img_files = img_file
        self.level = level
        self.totensor = None
        self.patch_size = patch_size
        self.device = device
        self.mask = None
        self.mask_level = level - 1

        image_reader = mir.MultiResolutionImageReader()
        self.image = image_reader.open(img_file)

        if mask_file is not None:
            mask_reader = mir.MultiResolutionImageReader()
            self.mask = mask_reader.open(mask_file)

        size = list(self.image.getLevelDimensions(level))
        self.downsampl = int(self.image.getLevelDownsample(level))

        self.mask_downsampl = int(self.mask.getLevelDownsample(self.mask_level))
        self.patch_nr = (-(size[0] // -patch_size[0]), -(size[1] // -patch_size[1]))

        if totensor:
            self.totensor = partial(augmentations.vaugment_fn, transform = augmentations.val_transforms)

    def __len__(self):
        return self.patch_nr[0] * self.patch_nr[1]

    def __getitem__(self, idx):
        n_y = idx // self.patch_nr[0]
        n_x = idx % self.patch_nr[0]

        x = self.patch_size[0] * n_x * self.downsampl
        y = self.patch_size[1] * n_y * self.downsampl
        patch = self.image.getUCharPatch(x, y, *self.patch_size, self.level)

        if self.mask is not None:
            x = self.patch_size[0] * n_x * self.mask_downsampl
            y = self.patch_size[1] * n_y * self.mask_downsampl
            mask = self.mask.getUCharPatch(x, y, *self.patch_size, self.mask_level)
        
        if not self.totensor and self.mask is not None:
            return patch, mask
        elif not self.totensor:
            return patch
        
        if self.mask is not None:
            patch, mask = self.totensor(patch, mask)
        else:
            patch, _ = self.totensor(patch, np.zeros_like((3,3,3)))

        patch = patch / 255.0

        if self.mask is not None:
            return patch.unsqueeze(0).to(self.device), mask.unsqueeze(0).to(self.device)
            
        return patch.unsqueeze(0).to(self.device)

    def getPatchNr(self):
        return self.patch_nr
    
class WSIMILset(Dataset):
    def __init__(self,
            config_file,
            partition,
            patch_shapes, 
            tissue_label=1,
            mask_spacing=None, 
            augmentations_pipeline=None, 
            iterations=0,
            spacing_tolerance=0.25,
            label_mode='load',
            cache_path=None,
            in_channels=None,
            target=None,
            task='classification',
            sample_size=1,
            clam_format=True
        ):
        self.config_file = config_file
        self.cache_path = cache_path
        self.partition = partition
                
        self.path_scores = None
        self.psource_list = None
        self.psampler_list = None

        self.tissue_label = tissue_label
        self.label_mode = label_mode
        self.target = ['bal', 'fib', 'sta', 'inf'] if target is None else target
        self.mask_spacing = list(patch_shapes.keys())[0] if mask_spacing is None else mask_spacing
        self.spacing_tolerance = spacing_tolerance

        self.spacing = list(patch_shapes.keys())[0]
        self.shape = list(patch_shapes.values())[0]
        self.in_channels = [0,1,2] if in_channels is None else in_channels
        
        self._aug_fn = augmentations_pipeline
        self.totensor = partial(augmentations.vaugment_fn, transform = augmentations.val_transforms)
        
        self.iterations = iterations
        self.n_imgs = 0

        self.task = task
        self.sample_size = sample_size

        self.clam_format = clam_format

    def step(self):
        # Initiate patch samplers
        self.psource_list, self.path_scores = CPS.parse_config(self.config_file, self.partition)
        self.n_imgs = len(self.path_scores)

        self.psampler_list = CPS.create_samplers(
            source_list=self.psource_list, 
            mask_spacing=self.mask_spacing, 
            spacing_tol=self.spacing_tolerance, 
            in_channels=self.in_channels, 
            label_mode=self.label_mode, 
            cache_path=self.cache_path)
        
        self.labels = []
        if self.task == 'classification':
            for path_score in self.path_scores:
                if self.clam_format:
                    path_score[0]  = path_score[0] - 1
                    path_score = path_score * 6
                self.labels.append(
                    self.transformLabels(torch.Tensor(path_score).to(torch.int64))
                )
        else:
            for path_score in self.path_scores:
                self.labels.append(torch.Tensor(path_score))
        
    def transformLabels(self, path_score):
        labels = []
        i = 0
        # if self.clam_format:
        #     return path_score #F.one_hot(path_score[0], num_classes=self.)
        if 'bal' in self.target:
            labels.append(F.one_hot(path_score[i], num_classes=3))
            i += 1
        if 'fib' in self.target:
            labels.append(F.one_hot(path_score[i], num_classes=5))
            i += 1
        if 'sta' in self.target:
            labels.append(F.one_hot(path_score[i], num_classes=4))
            i += 1
        if 'inf' in self.target:
            labels.append(F.one_hot(path_score[i], num_classes=3))
        if 'rnas' in self.target:
            labels.append(F.one_hot(path_score[i], num_classes=3))
        
        return torch.cat(labels, 0)
    
    def getSampleLabel(self):
        sampleLabel = self.labels[0]
        if self.task == 'classification':
            return self.transformLabels(torch.Tensor(sampleLabel).to(torch.int64))
        else:
            return torch.Tensor(sampleLabel)
     
    def __len__(self):
        return self.iterations
     
    def __getitem__(self, idx):
        item = idx % self.n_imgs
        
        patches, masks, labels = [], [], []
        for i in range(self.sample_size):
            patch, mask = self.psampler_list[item].single_sample(
                idx, 
                label=self.tissue_label, 
                shape=self.shape, 
                spacing=self.spacing)
            
            if self._aug_fn:
                patch, mask = self._aug_fn(patch, mask)
            else:
                patch, mask = self.totensor(patch, mask)
            
            label = self.labels[item]

            if self.sample_size != 1:
                patches.append(patch)
                masks.append(mask)
                labels.append(label)
        
        if self.sample_size != 1:
            patches = torch.Tensor(patches)
            masks = torch.Tensor(masks)
            labels = torch.Tensor(labels)
            return patches / 255.0, masks.float(), labels

        return patch / 255.0, mask.float(), label

class imagePatchSampler(Dataset):
    def __init__(self,
            config_file,
            partition='training',
            iterations=0,
            augmentations_pipeline=None, 
            n_classes=2,
            sample_size=1,
            simple_mode=True
            ):
        self.config_file = config_file
        
        self._aug_fn = augmentations_pipeline
        self.totensor = partial(augmentations.vaugment_fn, transform = augmentations.val_transforms)

        self.iterations = iterations
        self.n_classes = n_classes
        self.sample_size = sample_size
        self.partition = partition

        self.simple_mode = simple_mode
    
    def step(self):
        with open(self.config_file, 'r') as stream:
            data_loaded = yaml.safe_load(stream)
        
        root = data_loaded['path']['root']
        patches = data_loaded['path']['patches']

        self.samples = []
        self.labels = []
        for sample in data_loaded['data'][self.partition]['default']:
            image_dir = Path(Path(sample['image'].format(root=root, images=patches)).with_suffix(''))
            path_score = torch.Tensor(sample['path_scores']).to(torch.int64)
            
            sample_list = [i for i in image_dir.glob('*')]
            
            self.samples.append(sample_list)
            self.labels.append(F.one_hot(path_score[0] - 1, num_classes=self.n_classes))
            
        self.n_imgs = len(self.labels)

        if self.simple_mode:
            x = (self.iterations // len(self.samples))+1
            self.samples = self.samples * x 
            self.labels = self.labels * x

    def get_simple(self, item):
        n_samples = len(self.samples[item])
        i = np.random.choice(range(n_samples), 1)[0]
        
        patch = np.array(Image.open(self.samples[item][i]).convert('RGB'))

        label = self.labels[item]
            
        if self._aug_fn:
            patch, _ = self._aug_fn(patch, np.zeros((224,224,3)))
        else:
            patch, _ = self.totensor(patch, np.zeros((224,224,3)))

        return patch / 255.0, 0, label

    
    def __len__(self):
        if self.simple_mode:
            return len(self.samples)
        return self.iterations
    
    def __getitem__(self, idx):
        if self.simple_mode:
            return self.get_simple(idx)

        item = idx % self.n_imgs

        patches, labels = [], []
        n_samples = len(self.samples[item])

        for i in np.random.choice(range(n_samples), self.sample_size, replace=False):
            patch = np.array(Image.open(self.samples[item][i]).convert('RGB'))

            label = self.labels[item]
            
            if self._aug_fn:
                patch, _ = self._aug_fn(patch, np.zeros((224,224,3)))
            else:
                patch, _ = self.totensor(patch, np.zeros((224,224,3)))
            
            if self.sample_size != 1:
                patches.append(patch)
                labels.append(label)
        
        if self.sample_size != 1:
            patches = torch.stack(patches)
            labels = torch.stack(labels)
            return patches / 255.0, 0, labels

        return patch / 255.0, 0, label
    
    def getSampleLabel(self):
        sampleLabel = self.labels[0]
        return sampleLabel
    
class imagePatchSamplerInference:
    def __init__(self,
                 patch_dir):
        self.totensor = partial(augmentations.vaugment_fn, transform = augmentations.val_transforms)
        self.patch_dir = patch_dir
        return
    
    def step(self):
        self.sample_list = [i for i in self.patch_dir.glob('*.png')]
    
    def __len__(self):
        return len(self.sample_list)
    
    def __getitem__(self, idx):
        patch = np.array(Image.open(self.sample_list[idx]).convert('RGB'))
        patch, _ = self.totensor(patch, np.zeros(patch.shape))
        return patch / 255.0