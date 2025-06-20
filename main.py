import glob
import os

import hydra
import lightning as pl
import torch
import wandb
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.loggers import WandbLogger

from Dataloaders.TextTranslationDataloader import TextTranslationDataloader
from model.Transformer import Transformer
from model.TransformerLightning import TransformerLightning
from model.utils.utils import token_to_text

def get_newest_checkpoint(data_path):
    list_of_files = glob.glob(data_path + "/*")
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg):
    torch.manual_seed(cfg.seed)
    pl.seed_everything(cfg.seed, workers=True)
    torch.set_float32_matmul_precision('medium')

    print("Creating Dataloaders...")
    datamodule = TextTranslationDataloader(source_data_path=cfg.source_language.dataset_path,
                                           source_language=cfg.source_language.name,
                                           source_vocabulary_path=cfg.source_language.vocabulary_path,
                                           target_data_path=cfg.target_language.dataset_path,
                                           target_language=cfg.target_language.name,
                                           target_vocabulary_path=cfg.target_language.vocabulary_path,
                                           batch_size=cfg.batch_size)


    '''
        This variable is necessary to set the ignore index in the cross entropy loss 
    '''
    cfg.encoder.padding_idx = datamodule.dataset.source_tokenizer.encode("[PAD]").ids[0]
    cfg.decoder.padding_idx = datamodule.dataset.target_tokenizer.encode("[PAD]").ids[0]
    cfg.ignore_index = datamodule.dataset.target_tokenizer.encode("[PAD]").ids[0]

    print("Creating Model...")
    transformer = Transformer(cfg)
    model = TransformerLightning(torch_model=transformer, hparams=cfg)

    if cfg.from_checkpoint:
        model = TransformerLightning.load_from_checkpoint(get_newest_checkpoint(cfg.checkpoint_dir))

    logger = None
    if cfg.hpc:
        wandb.login(key=cfg.wandb.key)
        logger = WandbLogger(project=cfg.wandb.project_name, save_dir=cfg.logs_dir)
    else:
        logger = TensorBoardLogger(cfg.logs_dir)


    lr_monitor = LearningRateMonitor(logging_interval='step')
    model_checkpoint = ModelCheckpoint(monitor='val_loss_epoch',
                                            dirpath=cfg.checkpoint_dir,
                                       save_weights_only=True,
                                       save_top_k=1,
                                       save_last=True)

    trainer = pl.Trainer(max_epochs=cfg.max_epochs,
                         accelerator="gpu",
                         devices=1,
                         logger=logger,
                         profiler="simple",
                         callbacks=[model_checkpoint,
                                    lr_monitor])

    if cfg.train:
        print("Start Training...")
        trainer.fit(model, datamodule=datamodule)

    if cfg.test:
       pass





if __name__ == '__main__':
    main()