from typing import Any

import lightning as pl
import numpy as np
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import nn

from .Transformer import Transformer


class TransformerLightning(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters()
        self.model = Transformer(hparams)
        self.lr = hparams.lr
        self.training_loss_history = []
        self.validation_loss_history = []
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)
    def training_step(self, batch, batch_idx):
        source_batch, source_mask, target_batch, target_mask = batch
        x = self.model(x=source_batch,
                       y=target_batch,
                       source_mask=source_mask,
                       target_mask=target_mask)
        loss = self.loss(x.reshape(-1, x.shape[-1]), target_batch.reshape(-1))
        assert not torch.isnan(loss)
        self.log('train_loss', loss)
        self.training_loss_history.append(loss.cpu().detach().numpy())
        return loss

    def on_train_epoch_end(self) -> None:
        loss_mean = np.mean(self.training_loss_history)
        self.training_loss_history.clear()
        self.log("train_loss_epoch", loss_mean)

    def validation_step(self, batch, batch_idx):
        source_batch, source_mask, target_batch, target_mask = batch
        x = self.model(x=source_batch,
                       y=target_batch,
                       source_mask=source_mask,
                       target_mask=target_mask)
        loss = self.loss(x.reshape(-1, x.shape[-1]).float(), target_batch.reshape(-1))
        self.log('val_loss', loss)
        self.validation_loss_history.append(loss.cpu())
        return loss

    def on_validation_epoch_end(self) -> None:
        loss_mean = np.mean(self.validation_loss_history)
        self.validation_loss_history.clear()
        self.log("val_loss_epoch", loss_mean)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)


def main():
    pass

if __name__ == '__main__':
    main()