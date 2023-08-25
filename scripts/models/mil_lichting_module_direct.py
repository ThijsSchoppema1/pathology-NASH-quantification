# An lightning module to train in a MIL schema

import pytorch_lightning as pl
import torch.nn.functional as F
import torch
# from monai.metrics import compute_meandice as mean_dice
from torchmetrics.classification import MulticlassAUROC as AUROC
from torchmetrics.classification import MulticlassROC as ROC
# from torchmetrics.functional import dice_score
import torchmetrics as tm
# from pytorch_lightning.metrics.classification import AUROC 
from torch import optim
from torchmetrics.classification import MulticlassConfusionMatrix, MulticlassAccuracy

import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sn
import torchvision

class LitModelTrainer(pl.LightningModule):
    def __init__(self, model, loss_fn, lr, n_classes, patch_size, supervision, sample_size, selection_size, targets=["bal","fib","sta","inf"]):
        super().__init__()
        self.model = model
        self.loss = loss_fn[0]

        self.lr = lr
        self.n_classes = n_classes
        self.patch_size = patch_size
        self.supervision = supervision

        if selection_size is not None and selection_size > sample_size:
            selection_size = sample_size

        self.sample_size = sample_size
        self.selection_size = selection_size
        
        self.labelspc = []
        if "bal" in targets:
            self.labelspc.append(3)
        if "fib" in targets:
            self.labelspc.append(5)
        if "sta" in targets:
            self.labelspc.append(4)
        if "inf" in targets:
            self.labelspc.append(3)
            
        # Conf matrix
        self.valConf = [
            MulticlassConfusionMatrix(num_classes=nc, normalize='all').cpu() for nc in self.labelspc
        ]
        self.testConf = [
            MulticlassConfusionMatrix(num_classes=nc, normalize='all').cpu() for nc in self.labelspc
        ]

        self.valConfn = [
            MulticlassConfusionMatrix(num_classes=nc, normalize='true').cpu() for nc in self.labelspc
        ]
        self.testConfn = [
            MulticlassConfusionMatrix(num_classes=nc, normalize='true').cpu() for nc in self.labelspc
        ]
        
        if len(self.labelspc) == 1:
            self.valAcc = MulticlassAccuracy(num_classes=self.labelspc[0])
            self.testAcc = MulticlassAccuracy(num_classes=self.labelspc[0])

        # Labels
        self.labelList = ['Other', 'Staetosis', 'Inflammation']
        
        self.targets = targets

        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        inputs, _, targets = batch
        # Forward pass
        y_hat = self.model(inputs.squeeze())
        loss = self.loss(y_hat, targets.float())

        # Logging to TensorBoard (if installed) by default
        self.log_dict({"train_loss": loss}, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, _, targets = batch

        # Forward pass
        y_hat = self.model(inputs.squeeze())
        loss = self.loss(y_hat, targets.float())

        y_hat_s = F.softmax(y_hat)
        if len(self.labelspc) == 1:
            self.valAcc(y_hat_s, targets)
            self.log("val_acc", self.valAcc, prog_bar=True)

        c = 0
        for i in range(len(self.labelspc)):
                        
            self.valConf[i](y_hat_s[:, c:c+self.labelspc[i]].detach().cpu(), targets[:, c:c+self.labelspc[i]].long().detach().cpu())
            self.valConfn[i](y_hat_s[:, c:c+self.labelspc[i]].detach().cpu(), targets[:, c:c+self.labelspc[i]].long().detach().cpu())
            
            c += self.labelspc[i]

        self.log("val_loss", loss, prog_bar=True)
        # self.log("val_auc", self.valAuc, prog_bar=True)

        if (self.trainer.is_last_batch) and (self.trainer.current_epoch % 10 == 0):
            for i in range(len(self.labelspc)):
                computed_confusion = self.valConf[i].compute().detach().cpu().numpy()
                self.log_confusion_matrix(computed_confusion, f'val_{self.targets[i]}')

                computed_confusion = self.valConfn[i].compute().detach().cpu().numpy()
                self.log_confusion_matrix(computed_confusion, f'val_norm_{self.targets[i]}')

    def test_step(self, batch, batch_idx):
        # this is the test loop
        inputs, _, targets = batch

        # Forward pass
        y_hat = self.model(inputs)
        loss = self.loss(y_hat, targets.float())


        y_hat_s = F.softmax(y_hat)
        if len(self.labelspc) == 1:
            self.testAcc(y_hat_s, targets)
            
        
        c = 0
        for i in range(len(self.labelspc)):
                        
            self.testConf[i](y_hat_s[:, c:c+self.labelspc[i]].detach().cpu(), targets[:, c:c+self.labelspc[i]].long().detach().cpu())
            self.testConfn[i](y_hat_s[:, c:c+self.labelspc[i]].detach().cpu(), targets[:, c:c+self.labelspc[i]].long().detach().cpu())
            
            c += self.labelspc[i]

        self.log("test_loss", loss)

    def on_test_epoch_end(self):
        self.log("test_acc", self.testAcc)
        for i in range(len(self.labelspc)):
            computed_confusion = self.testConf[i].compute().detach().cpu().numpy()
            self.log_confusion_matrix(computed_confusion, f'test_{self.targets[i]}')

            computed_confusion = self.testConfn[i].compute().detach().cpu().numpy()
            self.log_confusion_matrix(computed_confusion, f'test_norm_{self.targets[i]}')

    def forward(self, x):
        return torch.argmax(self.model(x), dim=1)

    def configure_optimizers(self):
        opt = optim.Adam(self.model.parameters(), lr=self.lr) 
        scheduler = optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.1)
        return [opt], [scheduler]
    
    def get_model(self):
        return self.model
    
    def log_confusion_matrix(self, computed_confusion, loop_type='test'):
        df_cm = pd.DataFrame(
            computed_confusion
        )

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.subplots_adjust(left=0.05, right=.65)
        sn.set(font_scale=1.2)
        snax = sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt='.2f', ax=ax, xticklabels=self.labelList, yticklabels=self.labelList)
        snax.set(xlabel="Predictions", ylabel="True Labels")
        buf = io.BytesIO()

        plt.savefig(buf, format='jpeg', bbox_inches='tight')
        buf.seek(0)
        im = Image.open(buf)
        im = torchvision.transforms.ToTensor()(im)
        self.logger.experiment.add_image(f"{loop_type}_confusion_matrix", im, global_step=self.current_epoch)
        plt.close()

    def log_roc_curve(self, tpr, fpr, loop_type='test'):
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.subplots_adjust(left=0.05, right=.65)

        ax.plot(tpr, fpr, color='red')
        ax.axline( (0,0),slope=1,linestyle='--')
        ax.set_ylabel('FPR')
        ax.set_xlabel('TPR')
        ax.set_title('ROC curve')

        buf = io.BytesIO()

        plt.savefig(buf, format='jpeg', bbox_inches='tight')
        buf.seek(0)
        im = Image.open(buf)
        im = torchvision.transforms.ToTensor()(im)
        self.logger.experiment.add_image(f"{loop_type}_ROG", im, global_step=self.current_epoch)
        plt.close()
    
    def return_testConf(self):
        return self.testConf.compute().detach().cpu().numpy()


