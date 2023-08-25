# An empty lightning module to test the post-processing of the models

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
from torchmetrics.classification import MulticlassConfusionMatrix

import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sn
import torchvision

class LitModelTrainer(pl.LightningModule):
    def __init__(self, loss_fn, n_classes, labelNames=['Other', 'Staetosis', 'Inflammation']):
        super().__init__()
        self.loss = loss_fn[0]
        self.n_classes = n_classes

        # Dice scores
        self.trainDice = tm.Dice(average='micro', ignore_index=0)
        self.valDice = tm.Dice(average='micro', ignore_index=0)
        self.testDice = tm.Dice(average='micro', ignore_index=0)

        self.valMDice = tm.Dice(average='macro', ignore_index=0, num_classes=n_classes)
        self.testMDice = tm.Dice(average='macro', ignore_index=0, num_classes=n_classes)

        # self.valSDice = tm.Dice(average=None, ignore_index=0, num_classes=n_classes)
        # self.testSDice = tm.Dice(average=None, ignore_index=0, num_classes=n_classes)

        # Jaccard
        self.valJac = tm.JaccardIndex(task="multiclass", average='micro', ignore_index=0, num_classes=n_classes)
        self.testJac = tm.JaccardIndex(task="multiclass", average='micro', ignore_index=0, num_classes=n_classes)

        self.valMJac = tm.JaccardIndex(task="multiclass", average='macro', ignore_index=0, num_classes=n_classes)
        self.testMJac = tm.JaccardIndex(task="multiclass", average='macro', ignore_index=0, num_classes=n_classes)

        self.valSJac = tm.JaccardIndex(task="multiclass", average='none', ignore_index=0, num_classes=n_classes)
        self.testSJac = tm.JaccardIndex(task="multiclass", average='none', ignore_index=0, num_classes=n_classes)


        # AUROC Scores
        self.trainAuc = AUROC(num_classes=n_classes)
        self.valAuc = AUROC(num_classes=n_classes)
        self.testAuc = AUROC(num_classes=n_classes)

        # Conf matrix
        self.valConf = MulticlassConfusionMatrix(num_classes=n_classes, normalize='all')
        self.testConf = MulticlassConfusionMatrix(num_classes=n_classes, normalize='all')

        self.valConfn = MulticlassConfusionMatrix(num_classes=n_classes, normalize='true')
        self.testConfn = MulticlassConfusionMatrix(num_classes=n_classes, normalize='true')
        
        self.valConfs = MulticlassConfusionMatrix(num_classes=n_classes, normalize='none')
        self.testConfs = MulticlassConfusionMatrix(num_classes=n_classes, normalize='none')

        # ROC curve
        self.valRog = ROC(num_classes=n_classes)
        self.testRog = ROC(num_classes=n_classes)

        self.labelList = labelNames
            
        self.patch_vis = False
        self.test_mem = [[], []]

        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        y_hat, targets = batch
        y_hat[y_hat >= 10] = 4
        # y_hat = F.one_hot(y_hat, num_classes=self.n_classes)

        # Forward pass
        
        loss = self.loss(y_hat, targets.long())
        
        self.trainDice(y_hat, targets.long())
        self.trainAuc(y_hat, targets.long())

        # Logging to TensorBoard (if installed) by default
        self.log_dict({"train_loss": loss}, prog_bar=True)
        self.log("train_dice", self.trainDice)
        self.log("train_auc", self.trainAuc)
        
        self.patch_vis = True
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch

        y_hat[inputs >= 10] = 4
        # y_hat = F.one_hot(y_hat, num_classes=self.n_classes)
        # Forward pass
        
        loss = self.loss(y_hat, targets.long())
        y_hat, targets = y_hat.detach(), targets.detach()

        self.valDice(y_hat, targets.long())
        self.valAuc(y_hat, targets.long())
        self.valConf(y_hat, targets.long())
        self.valConfn(y_hat, targets.long())

        self.log("val_loss", loss)
        self.log("val_dice", self.valDice, prog_bar=True)
        self.log("val_auc", self.valAuc, prog_bar=True)

        if (self.patch_vis) and (self.trainer.current_epoch % 10 == 0):
            computed_confusion = self.valConf.compute().detach().cpu().numpy()
            self.log_confusion_matrix(computed_confusion, 'val')

            computed_confusion = self.valConfn.compute().detach().cpu().numpy()
            self.log_confusion_matrix(computed_confusion, 'val_norm')

            self.plot_patches_pred(inputs, y_hat, 'val')
            self.patch_vis = False
        
        self.valMDice(y_hat, targets.long())
        # self.valSDice(y_hat, targets.long())

        self.valJac(y_hat, targets.long())
        self.valMJac(y_hat, targets.long())
        self.valSJac(y_hat, targets.long())


        self.log("val_m_dice", self.valMDice)
        # self.log("val_s_dice", self.valSDice)

        self.log("val_jac", self.valJac)
        self.log("val_m_jac", self.valMJac)
        self.log("val_s_jac", self.valSJac)


    def test_step(self, batch, batch_idx):
        # this is the test loop
        y_hat, targets = batch
        y_hat = y_hat.squeeze(0)
        y_hat[y_hat >= 10] = 4
        # y_hat = F.one_hot(y_hat, num_classes=self.n_classes)

        # loss = self.loss(y_hat, targets.long())
        y_hat, targets = y_hat.detach(), targets.detach()

        # self.log("test_loss", loss)
        self.testDice(y_hat, targets.long())
        # self.testAuc(y_hat, targets.long())
        self.testConf(y_hat, targets.long())
        self.testConfn(y_hat, targets.long())
        self.testConfs(y_hat, targets.long())
        # self.testRog(y_hat, targets.long())

        self.testMDice(y_hat, targets.long())
        # self.testSDice(y_hat, targets.long())

        self.testJac(y_hat, targets.long())
        self.testMJac(y_hat, targets.long())
        self.testSJac(y_hat, targets.long())
    
    def on_test_epoch_end(self):

        self.log("test_dice", self.testDice)
        # self.log("test_auc", self.testAuc)
        computed_confusion = self.testConf.compute().detach().cpu().numpy()
        self.log_confusion_matrix(computed_confusion, 'test', title="Confusion matrix")

        computed_confusion = self.testConfn.compute().detach().cpu().numpy()
        self.log_confusion_matrix(computed_confusion, 'test_norm', title="Normalised confusion matrix")
        
        computed_confusion = self.testConfs.compute().detach().cpu().numpy()
        self.log_confusion_matrix(computed_confusion, 'test_abs', Abs=True, title="Absolute confusion matrix")

        # tpr, fpr, _ = self.testRog.compute()
        # self.log_roc_curve(tpr, fpr)
        self.log("test_m_dice", self.testMDice)
        # self.log("test_s_dice", self.testSDice)

        self.log("test_jac", self.testJac)
        self.log("test_m_jac", self.testMJac)

        for i, c in enumerate(self.testSJac.compute()):

            self.log(f"test_s{i}_jac", c)


    def forward(self, x):
        return torch.argmax(self.model(x), dim=1)

    def configure_optimizers(self):
        opt = optim.Adam(self.model.parameters(), lr=self.lr) 
        scheduler = optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.1)
        return [opt], [scheduler]
    
    def get_model(self):
        return self.model
    
    def log_confusion_matrix(self, computed_confusion, loop_type='test', Abs=False, title=''):
        df_cm = pd.DataFrame(
            computed_confusion
        )
        
        text_size = 14
        if Abs:
            text_size = 8

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.subplots_adjust(left=0.05, right=.65)
        sn.set(font_scale=1.2)
        snax = sn.heatmap(df_cm, annot=True, annot_kws={"size": text_size}, fmt='.2f', ax=ax, xticklabels=self.labelList, yticklabels=self.labelList)
        snax.set_xticklabels(snax.get_xticklabels(), rotation=0)
        snax.set(xlabel="Predictions", ylabel="True Labels")
        ax.set_title(title)
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
        
        if type(tpr) == list:
            for t, f in zip(tpr, fpr):
                ax.plot(t.detach().cpu().numpy(), f.detach().cpu().numpy())
        else:
            ax.plot(tpr.detach().cpu().numpy(), fpr.detach().cpu().numpy(), color='red')
        ax.axline( (0,0),slope=-1,linestyle='--')
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

    def plot_patches_pred(self, patches, preds, loop_type='test'):
        patches = patches.detach().cpu().numpy()
        preds = preds.detach().cpu()

        fig, axes = plt.subplots(4, 4, figsize=(20, 20))
        for i, (patch, pred) in enumerate(zip(patches, preds)):
            if len(pred) == 2:
                pred = pred[0]
            pred = torch.argmax(pred, dim=1).numpy()
            
            if i == 16:
                break
            axes[i // 4, i % 4].imshow(patch.T)

            pred = pred.squeeze()
            axes[i // 4, i % 4].imshow((pred.T)/self.n_classes, alpha = 0.5)
        
        buf = io.BytesIO()

        plt.savefig(buf, format='jpeg', bbox_inches='tight')
        buf.seek(0)
        im = Image.open(buf)
        im = torchvision.transforms.ToTensor()(im)
        self.logger.experiment.add_image(f"{loop_type}_preds", im, global_step=self.current_epoch)
        plt.close()
    
    def return_testConf(self):
        return self.testConf.compute().detach().cpu().numpy()


