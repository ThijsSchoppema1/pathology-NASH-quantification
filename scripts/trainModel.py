# Script to train a model

import click
from pathlib import Path
import yaml
import sys

# Sampling imports
import digitalpathology.generator.batch.simplesampler as sampler
from dataProcesses import augmentations, dataGenerators
import albumentations
from functools import partial
from torch.utils.data import DataLoader

# Model imports
from other import lossFunctions
import segmentation_models_pytorch as smp
from torchvision import models
from torch import nn

# Training imports
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

import os

SEED=42


@click.command()
@click.option('--config_file', type=Path, help='The config file for the training protocol, an example can be found in docs, train_config_example.yalm')
def main(
    config_file
    ):
    # Parse the given config file and prepare the data generators + model
    paths_p, sampler_p, train_p, model_p, test_model = parse_config(config_file)
    print("prepare data")
    train_gen, val_gen, test_gen, model_p = perpare_datageneration(sampler_p, paths_p, test_model, model_p, train_p)
    print("prepare model")
    model, loss_fn, pl_model = prepare_model(model_p, sampler_p, train_p)
    
    # Load train parameters
    batch_size = train_p['batch_size']
    epochs = train_p['epochs']
    accumulate_grad_batches = train_p['accumulate_grad_batches']

    out_path = Path(paths_p['out_path'])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('We are using the {} device.'.format(device))
    model = model.to(device)

    # set dataloaders and loggers
    train_dataloader = DataLoader(train_gen, batch_size=batch_size, shuffle=True, num_workers=8)
    val_dataloader = DataLoader(val_gen, batch_size=batch_size, shuffle=True, num_workers=8)
    
    checkpoint_callback = ModelCheckpoint(dirpath=out_path, 
                                          save_top_k=2,
                                        #   every_n_epochs=-(epochs // -5), # max 5 checkpoints
                                          monitor="val_loss",
                                          filename="unsupmodel_{epoch:02d}_{val_loss:.2f}",
                                          save_last=True)
    train_logger = pl_loggers.TensorBoardLogger(save_dir=out_path / "logs")

    # Define PL trainer
    trainer = pl.Trainer(max_epochs=epochs,
                          accelerator='gpu', 
                          devices=-1, 
                          callbacks=[checkpoint_callback], 
                          logger=train_logger, 
                          accumulate_grad_batches=accumulate_grad_batches)#,
                        #   gradient_clip_val=0.5)
    
    # Check if last checkpoint exists
    resume_checkpoint = paths_p['resume_checkpoint']
    if resume_checkpoint is not None:
        resume_checkpoint = Path(resume_checkpoint)
    if (out_path / 'last.ckpt').is_file():
        resume_checkpoint=out_path
    
    # Train model or resume training
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
    
    # Test model
    if test_model:
        test_dataloader = DataLoader(test_gen, batch_size=1, shuffle=False)
        trainer.test(pl_model, dataloaders=test_dataloader)

    if model_p['model_type'] == 'wnet':
        from models.lichting_module import LitModelTrainer
    else:
        from models.sup_lichting_module import LitModelTrainer

    torch.save(pl_model.get_model().state_dict(), os.path.join(out_path, 'model_last_statedict' + '.pt'))
    pl_model = LitModelTrainer.load_from_checkpoint(checkpoint_path=checkpoint_callback.best_model_path)
    torch.save(pl_model.get_model().state_dict(), os.path.join(out_path, 'model_best_statedict' + '.pt'))

def perpare_datageneration(sampler_params, paths_p, test_model, model_p, train_p, post_test=False):
    # Select general params
    task = sampler_params['task']
    patch_shapes = sampler_params['patch_shapes']
    iterations = sampler_params['iterations']
    mode = sampler_params['mode']
    mask_spacing = sampler_params['mask_spacing']
        
    source_file = paths_p['source_file']
    
    # Define augmentations
    transforms = augmentations.transforms
    albumentations.save(transforms, "./albumentations.yaml", data_format='yaml')
    aug_partial = partial(augmentations.augment_fn, transform=transforms)

    test_gen = None
    # Load samplers for each noted task
    if task == 'segmentation':
        label_dist = sampler_params['label_dist']
        label_map = sampler_params['label_map']
        sampler_count = sampler_params['sampler_count']
        sselect = sampler_params['strict_selection']

        train_gen = None
        # Skip if testing the post-processing
        if not post_test:
            training_sampler = sampler.SimpleSampler(patch_source_filepath=source_file, partition='training', label_dist=label_dist, label_map=label_map,
                                            label_mode=mode, patch_shapes=patch_shapes, mask_spacing=mask_spacing, 
                                            sampler_count=sampler_count, iterations=iterations, seed=SEED, strict_selection=sselect)
            training_sampler.step()

            train_gen = dataGenerators.UnSupDataset(training_sampler, augmentations_pipeline=aug_partial, toFloat=False)

        input_channels=None
        tF = False
        # Change channels if post-processing
        if post_test:
            input_channels=[0]
            tF = post_test

        val_sampler = sampler.SimpleSampler(patch_source_filepath=source_file, partition='validation', label_dist=label_dist, label_map=label_map,
                                label_mode=mode, patch_shapes=patch_shapes, mask_spacing=mask_spacing, 
                                sampler_count=sampler_count, iterations=iterations, seed=SEED, strict_selection=sselect, input_channels=input_channels)
        val_sampler.step()
        val_gen = dataGenerators.UnSupDataset(val_sampler, toFloat=post_test)

        if test_model:
            test_sampler = sampler.SimpleSampler(patch_source_filepath=source_file, partition='testing', label_dist=label_dist, label_map=label_map,
                                         label_mode=mode, patch_shapes=patch_shapes, mask_spacing=mask_spacing, 
                                         sampler_count=sampler_count, iterations=iterations, seed=SEED, input_channels=input_channels)
            test_sampler.step()
            test_gen = dataGenerators.UnSupDataset(test_sampler, toFloat=tF)

    elif task == 'classification' or task == 'regression':
        targets = model_p['targets']
        sample_size=train_p['sample_size']
        
        train_gen = dataGenerators.WSIMILset(config_file=source_file, partition='training', label_mode=mode, patch_shapes=patch_shapes,
                                  mask_spacing=mask_spacing, iterations=iterations, augmentations_pipeline=aug_partial, target=targets, sample_size=sample_size)
        val_gen = dataGenerators.WSIMILset(config_file=source_file, partition='validation', label_mode=mode, patch_shapes=patch_shapes,
                                  mask_spacing=mask_spacing, iterations=iterations, target=targets, sample_size=sample_size)
        
        train_gen.step()
        val_gen.step()

        if test_model:
            test_gen = dataGenerators.WSIMILset(config_file=source_file, partition='testing', label_mode=mode, patch_shapes=patch_shapes,
                                  mask_spacing=mask_spacing, iterations=iterations, target=targets)
            test_gen.step()
        
        model_p['n_classes'] = len(train_gen.getSampleLabel())
    
    elif task == 'classificationPatch' or task == 'classificationSimplePatch':
        mode = False
        if task == 'classificationSimplePatch':
            mode = True

        targets = model_p['targets']
        n_classes = model_p['n_classes']
        sample_size=train_p['sample_size']
        
        train_gen = dataGenerators.imagePatchSampler(config_file=source_file, partition='training', simple_mode=mode,
                                  iterations=iterations, augmentations_pipeline=aug_partial, n_classes=n_classes, sample_size=sample_size)
        val_gen = dataGenerators.imagePatchSampler(config_file=source_file, partition='validation', simple_mode=mode,
                                  iterations=iterations, n_classes=n_classes, sample_size=sample_size)
        
        train_gen.step()
        val_gen.step()

        if test_model:
            test_gen = dataGenerators.imagePatchSampler(config_file=source_file, partition='testing', simple_mode=mode,
                                  iterations=iterations, n_classes=n_classes, sample_size=sample_size)
            test_gen.step()
        
        model_p['n_classes'] = len(train_gen.getSampleLabel())
        
    else:
        print(f'task {task} not found.')
        sys.exit(0)
    
    return train_gen, val_gen, test_gen, model_p

def prepare_model(model_p, sampler_p, train_p, testModelCall=False):
    loss = model_p['loss']
    model_type = model_p['model_type']
    dropout = model_p['dropout']
    n_classes = model_p['n_classes']
    pretrained_weights = model_p['weights']
    lweights = model_p['lweights']
    lr = train_p['lr']

    patch_size = list(sampler_p['patch_shapes'].values())[0]

    if loss == 'supervisedLoss':
        loss_fn = lossFunctions.getSupervisedLoss(lweights=lweights)
    elif loss == 'unsupervisedLoss':
        loss_fn = lossFunctions.sep_losses(lweights=lweights)
    else:
        print(f'loss function {loss} not found')
        sys.exit(0)
    
    if model_type == 'unet':
        from models.sup_lichting_module import LitModelTrainer
        encoder = model_p['encoder_name']
        labelNames = model_p['label_names']
        model = smp.Unet(encoder_name=encoder, classes=n_classes, encoder_weights=pretrained_weights, activation=None)
        pl_model = LitModelTrainer(model, loss_fn, lr, n_classes, patch_size, supervision=True, labelNames=labelNames)
    elif model_type == 'pointwiseunet':
        from models.pointwise_u_net import UNet
        from models.sup_lichting_module import LitModelTrainer
        model = UNet(3, n_classes, 'batch_norm', dropout)
        pl_model = LitModelTrainer(model, loss_fn, lr, n_classes, patch_size, supervision=True)
    elif model_type == 'wnet':
        from models.lichting_module import LitModelTrainer
        from models.w_net2 import WNet
        model = WNet(3, n_classes, "batch_norm", dropout)
        pl_model = LitModelTrainer(model, loss_fn, lr, n_classes, patch_size, supervision=True)
    elif model_type == 'resnet50':
        if sampler_p['task'] == 'classificationSimplePatch':
            from models.mil_lichting_module_direct import LitModelTrainer
            loss_fn = [nn.CrossEntropyLoss()]
        else:
            from models.mil_lichting_module import LitModelTrainer
            loss_fn = [nn.CrossEntropyLoss(reduce=False)]
        sample_size = train_p['sample_size']
        selection_size = train_p['selection_size']
        targets = model_p['targets']
                
        model = models.resnet50(weights="IMAGENET1K_V2")
        model.fc = nn.Linear(2048, n_classes)
        
        pl_model = LitModelTrainer(model, loss_fn, lr, n_classes, patch_size, True, sample_size, selection_size, targets)
    elif model_type == 'efficientnet-b4':
        if sampler_p['task'] == 'classificationSimplePatch':
            from models.mil_lichting_module_direct import LitModelTrainer
            loss_fn = [nn.CrossEntropyLoss()]
        else:
            from models.mil_lichting_module import LitModelTrainer
            loss_fn = [nn.CrossEntropyLoss(reduce=False)]
        sample_size = train_p['sample_size']
        selection_size = train_p['selection_size']
        targets = model_p['targets']
                
        model = models.efficientnet_b4(weights="IMAGENET1K_V1")
        model.fc = nn.Linear(512, n_classes)
        pl_model = LitModelTrainer(model, loss_fn, lr, n_classes, patch_size, True, sample_size, selection_size, targets)
    elif model_type == 'empty':
        from models.empty_module import LitModelTrainer
        labelNames = model_p['label_names']
        model = None
        pl_model = LitModelTrainer(loss_fn, n_classes, labelNames)
    else:
        print(f'model type {model_type} not found')
        sys.exit(0)

    if testModelCall:
        return model, loss_fn, LitModelTrainer

    return model, loss_fn, pl_model

def parse_config(config_file):
    with open(config_file, 'r') as stream:
        data_loaded = yaml.safe_load(stream)
    paths_p = data_loaded['paths']
    sampler_p = data_loaded['sampler']
    train_p = data_loaded['training']
    model_p = data_loaded['model']
    test_model = data_loaded['test_model']
    return paths_p, sampler_p, train_p, model_p, test_model

if __name__ == "__main__":
    main()
