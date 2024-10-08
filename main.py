import glob
import os

import hydra
import lightning as pl
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from Dataloaders.TextTranslationDataloader import TextTranslationDataloader
from model.TransformerLightning import TransformerLightning

def get_newest_checkpoint(data_path):
    list_of_files = glob.glob(data_path + "/*")
    latest_file = max(list_of_files, key=os.path.getctime)
    print(latest_file)
    return latest_file
@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg):
    torch.set_float32_matmul_precision('medium')
    datamodule = TextTranslationDataloader(source_data_path=cfg.source_language.dataset_path,
                                           source_language=cfg.source_language.name,
                                           target_data_path=cfg.target_language.dataset_path,
                                           target_language=cfg.target_language.name,
                                           batch_size=cfg.batch_size,
                                           max_dataset_lenght=cfg.maximum_dataset_lenght)

    cfg.encoder.vocab_size = datamodule.dataset.source_vocab_size
    cfg.decoder.vocab_size = datamodule.dataset.target_vocab_size
    model = None
    if cfg.from_checkpoint:
        model = TransformerLightning.load_from_checkpoint(get_newest_checkpoint(cfg.checkpoint_dir))
    else:
        model = TransformerLightning(cfg)

    logger = TensorBoardLogger(cfg.logs_dir)
    trainer = pl.Trainer(max_epochs=cfg.max_epochs,
                         accelerator="gpu",
                         devices=1,
                         logger=logger,
                         callbacks=[ModelCheckpoint(monitor='val_loss_epoch',
                                                    dirpath=cfg.checkpoint_dir,
                                                    )])

    if cfg.train:
        trainer.fit(model, datamodule=datamodule)

    if cfg.test:
        model.eval()
        prompt = "Hi, How are you?"
        tokenized_sentence = torch.LongTensor([t.idx for t in datamodule.dataset.source_tokenizer(prompt)])

        out = model(tokenized_sentence)

if __name__ == '__main__':
    main()