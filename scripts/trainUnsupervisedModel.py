# Deprecated, bet tus use trainModel.py

import digitalpathology.generator.batch.simplesampler as sampler

from models.lichting_module import LitModelTrainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from other import lossFunctions
from dataProcesses import augmentations, dataGenerators

import albumentations
from functools import partial

import torch.optim as optim 
import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

import time
import click
import os

from pathlib import Path

@click.command()
@click.option('--source_file', type=str, default="./training_data_config.yaml")
@click.option('--patch_size', type=int, default=512)
@click.option('--level', type=int, default=0)
@click.option('--spacing', type=float, default=2.0)
@click.option('--iterations', type=int, default=1000)
@click.option('--batch_size', type=int, default=8)
@click.option('--n_classes', type=int, default=5)
@click.option('--dropout', type=float, default=0.3)
@click.option('--lr', type=float, default=0.0005)
@click.option('--out_path', type=Path)
@click.option('--epochs', type=int)
@click.option('--accumulate_grad_batches', type=int, default=1)
@click.option('--model_type', type=str, default='wnet')
@click.option('--supervision', type=bool, default=False)
@click.option('--resume_checkpoint', type=Path, default=None)
@click.option('--test_model', type=bool, default=False)
def main(
    source_file,
    patch_size,
    level,
    spacing,
    iterations,
    batch_size,
    n_classes,
    dropout,
    lr,
    out_path,
    epochs,
    accumulate_grad_batches,
    model_type,
    supervision,
    resume_checkpoint,
    test_model
    ):

    # Parameters
    patch_shapes = {level: [patch_size,patch_size]}
    label_dist = {1: 1.0} # sampling ratios per label
    val_label_dist = {1: 1.0} #, 2:1.0, 3:1.0, 4:1.0} # sampling ratios per label
    sampler_count=2 # amount of images to prepare
    seed = 42 # set the seed if you want your sampling to be deterministic
    mode = 'load'

    # Prepare data samplers
    training_sampler = sampler.SimpleSampler(patch_source_filepath=source_file, partition='training', label_dist=label_dist, 
                                         label_mode=mode, patch_shapes=patch_shapes, mask_spacing=spacing, 
                                         sampler_count=sampler_count, iterations=iterations, seed=seed)

    val_sampler = sampler.SimpleSampler(patch_source_filepath=source_file, partition='validation', label_dist=val_label_dist, 
                                        label_mode=mode, patch_shapes=patch_shapes, mask_spacing=spacing, 
                                        sampler_count=sampler_count, iterations=iterations, seed=seed)

    training_sampler.step()
    val_sampler.step()

    # Define Model
    if model_type == 'wnet':
        from models.w_net import WNet
        model = WNet(
            3,
            n_classes,
            "batch_norm",
            dropout,
        )
        loss_fn = lossFunctions.Wnet_loss
    elif model_type == 'wnet2':
        from models.w_net2 import WNet
        model = WNet(
            3,
            n_classes,
            "batch_norm",
            dropout,
        )
        loss_fn = lossFunctions.sep_losses()
    else:
        print(f'model type {model_type} not found')
        return 1

    # Define augmentations
    transforms = augmentations.transforms
    albumentations.save(transforms, "./albumentations.yaml", data_format='yaml')
    aug_partial = partial(augmentations.augment_fn, transform=transforms)

    # Datagenerator
    training_gen = dataGenerators.UnSupDataset(training_sampler, augmentations_pipeline=aug_partial)
    val_gen = dataGenerators.UnSupDataset(val_sampler)
    
    # Train Loop
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('We are using the {} device.'.format(device))
    model = model.to(device)

    train_dataloader = DataLoader(training_gen, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_gen, batch_size=batch_size, shuffle=True)
    
    checkpoint_callback = ModelCheckpoint(dirpath=out_path, 
                                          save_top_k=2,
                                        #   every_n_epochs=-(epochs // -5), # max 5 checkpoints
                                          monitor="val_loss",
                                          filename="unsupmodel_{epoch:02d}_{val_loss:.2f}",
                                          save_last=True)
    train_logger = pl_loggers.TensorBoardLogger(save_dir=out_path / "logs")

    pl_model = LitModelTrainer(model, loss_fn, lr, n_classes, patch_size, supervision)
    trainer = pl.Trainer(max_epochs=epochs,
                          accelerator='gpu', 
                          devices=-1, 
                          callbacks=[checkpoint_callback], 
                          logger=train_logger, 
                          accumulate_grad_batches=accumulate_grad_batches)
    
    # Check if last checkpoint exists
    if (out_path / 'last.ckpt').is_file():
        resume_checkpoint=out_path
    
    if resume_checkpoint is not None:
        trainer.fit(pl_model,
            train_dataloader,
            val_dataloader,
            ckpt_path=resume_checkpoint / 'last.ckpt'
            # precision=16, # <-- if model is to large or takes to long,
            )
    else:
        trainer.fit(pl_model,
                    train_dataloader,
                    val_dataloader,
                    # precision=16, # <-- if model is to large or takes to long,
                    )
    
    if test_model:
        test_sampler = sampler.SimpleSampler(patch_source_filepath=source_file, partition='testing', label_dist=val_label_dist, 
                                        label_mode=mode, patch_shapes=patch_shapes, mask_spacing=spacing, 
                                        sampler_count=sampler_count, iterations=iterations, seed=seed)
        test_sampler.step()
        test_gen = dataGenerators.UnSupDataset(test_sampler)
        test_dataloader = DataLoader(test_gen, batch_size=1, shuffle=False)

        trainer.test(model, dataloaders=test_dataloader)

    torch.save(pl_model.get_model().state_dict(), os.path.join(out_path, 'model_last_statedict' + '.pt'))
    pl_model = LitModelTrainer.load_from_checkpoint(checkpoint_path=checkpoint_callback.best_model_path)
    torch.save(pl_model.get_model().state_dict(), os.path.join(out_path, 'model_best_statedict' + '.pt'))

if __name__ == "__main__":
    main()