# Augmentations used by training

from albumentations.pytorch import ToTensorV2
from albumentations import (
    Compose, RandomBrightness, JpegCompression, HueSaturationValue, RandomContrast, HorizontalFlip,
    VerticalFlip, ElasticTransform
)

transforms = Compose([
        RandomBrightness(limit=0.5, p=0.2),
        JpegCompression(quality_lower=85, quality_upper=100, p=0.15),
        HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.2),
        RandomContrast(limit=0.2, p=0.2),
        HorizontalFlip(p=0.5),
        ElasticTransform(alpha=25, p=0.15),
        VerticalFlip(p=0.5),
        ToTensorV2()
    ])

val_transforms = Compose([
    ToTensorV2()
])

def augment_fn(patch, mask, transform):
    transformed = transform(image=patch, mask=mask)
    return transformed["image"], transformed["mask"]

def vaugment_fn(patch, mask, transform):
    transformed = val_transforms(image=patch, mask=mask)
    return transformed["image"], transformed["mask"]