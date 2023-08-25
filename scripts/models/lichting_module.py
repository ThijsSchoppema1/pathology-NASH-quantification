# An empty lightning module to train in an unsupervised manner for the W-net

import pytorch_lightning as pl
import torch
from torch import optim

import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sn
import torchvision
import torchmetrics as tm
from torchmetrics.classification import MulticlassAUROC as AUROC, MulticlassConfusionMatrix

class LitModelTrainer(pl.LightningModule):
    def __init__(self, model, loss_fn, lr, n_classes, patch_size, supervision):
        super().__init__()
        self.model = model
        self.loss_scl = loss_fn[0]
        self.loss_r = loss_fn[1]

        self.lr = lr
        self.n_classes = n_classes
        self.patch_size = patch_size
        self.supervision = supervision

        self.testDice = tm.Dice(average='micro', ignore_index=0)
        self.testAuc = AUROC(num_classes=n_classes)
        self.testConf = MulticlassConfusionMatrix(num_classes=n_classes, normalize='true')

        self.automatic_optimization = False
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        # Get optimizers
        scl_opt, r_opt = self.optimizers()
        inputs, _ = batch

        # First pass
        segmentation = self.model.forward_encoder(inputs.squeeze())
        scl_loss = self.loss_scl(inputs, segmentation)
        
        scl_opt.zero_grad()
        self.manual_backward(scl_loss)
        scl_opt.step()

        # Second pass
        segmentation, reconstruction = self.model(inputs.squeeze())
        r_loss = self.loss_r(inputs, reconstruction)

        r_opt.zero_grad()
        self.manual_backward(r_loss)
        r_opt.step()

        # Logging to TensorBoard (if installed) by default
        self.log_dict({"train_scl_loss": scl_loss, "train_r_loss": r_loss}, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        # Forward pass
        inputs, _ = batch
        segmentation, reconstruction = self.model(inputs.squeeze())

        scl_loss = self.loss_scl(inputs, segmentation)
        r_loss = self.loss_r(inputs, reconstruction)     

        # Logging to TensorBoard (if installed) by default
        self.log("val_loss", scl_loss + r_loss)
        self.log("val_scl_loss", scl_loss)
        self.log("val_r_loss", r_loss)
    
    def test_step(self, batch, batch_idx):
        # this is the test loop
        inputs, targets = batch

        segmentation, reconstruction = self.model(inputs.squeeze())
        scl_loss = self.loss_scl(inputs, segmentation)
        r_loss = self.loss_r(inputs, reconstruction)  

        self.log("test_loss", scl_loss + r_loss)
        self.log("test_scl_loss", scl_loss)
        self.log("test_r_loss", r_loss)

    def forward(self, x):
        return torch.argmax(self.model.forward_encoder(x), dim=1)

    def configure_optimizers(self):
        scl_opt = optim.SGD(self.model.U_enc.parameters(), lr=self.lr) 
        r_opt = optim.SGD(self.model.parameters(), lr=self.lr) 
        return scl_opt, r_opt
    
    def get_model(self):
        return self.model
    
    def setup_inference(self, class_conv=None):
        self.model.remove_dec()

        if class_conv:
            self.class_conv = class_conv

    def log_confusion_matrix(self, computed_confusion, loop_type='test'):
        df_cm = pd.DataFrame(
            computed_confusion
        )

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.subplots_adjust(left=0.05, right=.65)
        sn.set(font_scale=1.2)
        sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt='.2f', ax=ax)
        buf = io.BytesIO()

        plt.savefig(buf, format='jpeg', bbox_inches='tight')
        buf.seek(0)
        im = Image.open(buf)
        im = torchvision.transforms.ToTensor()(im)
        self.logger.experiment.add_image(f"{loop_type}_confusion_matrix", im, global_step=self.current_epoch)
    
    def return_testConf(self):
        return self.testConf.compute().detach().cpu().numpy()