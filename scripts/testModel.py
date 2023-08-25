# Script to test a trained model or post-processinb pipeline

import click
from pathlib import Path

from trainModel import parse_config, prepare_model, perpare_datageneration

import torch
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import seaborn as sn
import json
import pandas as pd

SEED=42

@click.command()
@click.option('-c', '--config_file', type=Path, help='The config file for the training protocol, an example can be found in docs, train_config_example.yalm')
@click.option('-s', '--ckpt_file_stem', type=str, default='', help="The name of the ckpt file, model_best_statedict.pt or model_last_staedict.pt are most used.")
@click.option('-i', '--class_conversion', type=dict, default=None, help="How to convert the classes, only usable for wnet, other models use the config file.")
@click.option('-k', '--convert_model', type=bool, default=False, help="Convert the model to an onnx file, deprecated.")
@click.option('-p', '--post_model', type=bool, default=False, help="Test the results of the post-processing")
@click.option('-v', '--val_set', type=bool, default=False, help="Test the validation set")
def main(
    config_file,
    ckpt_file_stem,
    class_conversion,
    convert_model,
    post_model,
    val_set
    ):
    paths_p, sampler_p, train_p, model_p, test_model = parse_config(config_file)
    _, val_gen, test_gen, model_p = perpare_datageneration(sampler_p, paths_p, test_model, model_p, train_p, post_model)
    

    out_path = Path(paths_p['out_path'])

    modelpath = out_path / ckpt_file_stem
    model, loss_fn, LitModelTrainer = prepare_model(model_p, sampler_p, train_p)

    if '.ckpt' == modelpath.suffix and model is not None:
        pl_model = LitModelTrainer.load_from_checkpoint(modelpath)
    elif '.pt' == modelpath.suffix and model is not None:
        model.load_state_dict(torch.load(modelpath))
        lr = train_p['lr']
        n_classes = model_p['n_classes']
        patch_size = list(sampler_p['patch_shapes'].values())[0]
        LitModelTrainer.set_model(model)
        pl_model = LitModelTrainer
    
    if model_p['model_type'] == 'wnet' and model is not None:
        pl_model.setup_inference(class_conversion)
        
    if post_model:
        pl_model = LitModelTrainer


    batch_size = train_p['batch_size']
    epochs = train_p['epochs']
    accumulate_grad_batches = train_p['accumulate_grad_batches']


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('We are using the {} device.'.format(device))

    checkpoint_callback = ModelCheckpoint(dirpath=out_path, 
                                          save_top_k=2,
                                        #   every_n_epochs=-(epochs // -5), # max 5 checkpoints
                                          monitor="val_loss",
                                          filename="unsupmodel_{epoch:02d}_{val_loss:.2f}",
                                          save_last=True)
    if not val_set:
        train_logger = pl_loggers.TensorBoardLogger(save_dir=out_path / "test_logs")
    else:
        train_logger = pl_loggers.TensorBoardLogger(save_dir=out_path / "val_logs")

    if not post_model: 
        trainer = pl.Trainer(max_epochs=epochs,
                            accelerator='gpu', 
                            devices=-1, 
                            callbacks=[checkpoint_callback], 
                            logger=train_logger, 
                            accumulate_grad_batches=accumulate_grad_batches)
    else:
        trainer = pl.Trainer(max_epochs=epochs,
                            accelerator='gpu', 
                            callbacks=[checkpoint_callback], 
                            logger=train_logger, 
                            accumulate_grad_batches=accumulate_grad_batches)
    
    if not val_set:
        test_dataloader = DataLoader(test_gen, batch_size=1, shuffle=False)
        trainer.test(pl_model, dataloaders=test_dataloader)
    else:
        test_dataloader = DataLoader(val_gen, batch_size=1, shuffle=False)
        trainer.test(pl_model, dataloaders=test_dataloader)

    testconf = pl_model.return_testConf()

    df_cm = pd.DataFrame(
        testconf
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.subplots_adjust(left=0.05, right=.65)
    sn.set(font_scale=1.2)
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt='.2f', ax=ax)

    plt.savefig(out_path / 'Test_conf_matrix.jpeg', format='jpeg', bbox_inches='tight')

    logged_metrics = trainer.logged_metrics
    for key in logged_metrics:
        logged_metrics[key] = logged_metrics[key].detach().cpu().tolist()
    with open(out_path / 'test_metrics.json', 'w') as f:
        json.dump(logged_metrics, f)

    if convert_model and False:

        model = pl_model.get_model()

        if model_p['model_type'] == 'wnet':
            model = model.return_enc()

        dummy_input = torch.randn(10, 3, 224, 224, device="cuda")
        torch.onnx.export(model, dummy_input, out_path / "output_model.onnx", verbose=True)


if __name__ == "__main__":
    main()