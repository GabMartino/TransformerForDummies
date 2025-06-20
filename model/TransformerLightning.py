from typing import Any, Optional

import lightning as pl
import numpy as np
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT, LRSchedulerTypeUnion
from sympy.physics.units import frequency
from timm.scheduler import StepLRScheduler
from torch import nn
from torch.optim.lr_scheduler import LambdaLR

from .Transformer import Transformer


class TransformerLightning(pl.LightningModule):
    def __init__(self, torch_model, hparams):
        super().__init__()
        self.save_hyperparameters()
        self.lr = hparams.lr
        self.embedding_size = hparams.embedding_size
        self.warmup_steps = hparams.warmup_steps
        self.with_scheduler = hparams.with_scheduler

        self.model = torch_model
        self.training_loss_history = []
        self.validation_loss_history = []
        '''
            The cross entropy loss has the ignore_index parameter.
            We can use it to avoid the loss to be computed for the padding values.
            Unfortunately in our implementation the padding strictly depend on the dataset so is passed as parameter
        '''
        self.ignore_index = hparams.ignore_index
        self.loss = nn.CrossEntropyLoss(ignore_index=self.ignore_index, label_smoothing=hparams.label_smoothing)


    def training_step(self, batch, batch_idx):
        source_batch, source_mask, target_batch, target_mask = batch
        x = self.model(encoder_input=source_batch,
                       decoder_input=target_batch,
                       source_padding_mask_keys=source_mask,
                       target_padding_mask_keys=target_mask)
        '''
            To implement the shift right:
            1. Roll the target batch to the left 
        '''
        target_batch_out = torch.roll(target_batch, -1, dims=-1)
        '''
            2. Assign to the last element of the target batch (that before was the first) the ignore index
        '''
        target_batch_out[:, -1] = self.ignore_index
        target_batch_out = target_batch_out.reshape(-1)
        loss = self.loss(x.reshape(-1, x.shape[-1]), target_batch_out)
        self.log('train_loss', loss)
        self.training_loss_history.append(loss.cpu().detach().numpy())
        return loss

    def on_train_epoch_end(self) -> None:
        loss_mean = np.mean(self.training_loss_history)
        self.training_loss_history.clear()
        self.log("train_loss_epoch", loss_mean)

    def validation_step(self, batch, batch_idx):
        source_batch, source_mask, target_batch, target_mask = batch
        x = self.model(encoder_input=source_batch,
                       decoder_input=target_batch,
                       source_padding_mask_keys=source_mask,
                       target_padding_mask_keys=target_mask)
        target_batch_out = torch.roll(target_batch, -1, dims=-1)
        target_batch_out[:, -1] = self.ignore_index
        target_batch_out = target_batch_out.reshape(-1)
        loss = self.loss(x.reshape(-1, x.shape[-1]), target_batch_out)
        self.log('val_loss', loss)
        self.validation_loss_history.append(loss.cpu())
        return loss

    def on_validation_epoch_end(self) -> None:
        loss_mean = np.mean(self.validation_loss_history)
        self.validation_loss_history.clear()
        self.log("val_loss_epoch", loss_mean)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.98), eps=1e-9)
        if self.with_scheduler:
            lr_scheduler = lambda _ : (((1 / self.embedding_size** 0.5)* min( 1/(1 + self.global_step ** 0.5), \
                (1 + self.global_step) * (1/ self.warmup_steps** 1.5))) / (self.current_epoch + 1))

            scheduler = LambdaLR(optimizer, lr_lambda=lr_scheduler )
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
        else:
            return optimizer


