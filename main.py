import glob
import os

import hydra
import lightning as pl
import torch
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from torch.distributions import OneHotCategorical, Categorical

from Dataloaders.TextTranslationDataloader import TextTranslationDataloader
from model.TransformerLightning import TransformerLightning

def get_newest_checkpoint(data_path):
    list_of_files = glob.glob(data_path + "/*")
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg):
    torch.manual_seed(cfg.seed)
    pl.seed_everything(cfg.seed)
    torch.set_float32_matmul_precision('medium')
    datamodule = TextTranslationDataloader(source_data_path=cfg.source_language.dataset_path,
                                           source_language=cfg.source_language.name,
                                           target_data_path=cfg.target_language.dataset_path,
                                           target_language=cfg.target_language.name,
                                           batch_size=cfg.batch_size,
                                           max_dataset_lenght=cfg.maximum_dataset_lenght)

    cfg.encoder.vocab_size = datamodule.dataset.source_vocab_size
    cfg.decoder.vocab_size = datamodule.dataset.target_vocab_size
    cfg.ignore_index = datamodule.dataset.target_special_characters['<PAD>']
    model = None
    if cfg.from_checkpoint:
        model = TransformerLightning.load_from_checkpoint(get_newest_checkpoint(cfg.checkpoint_dir))
    else:
        model = TransformerLightning(cfg)

    logger = TensorBoardLogger(cfg.logs_dir)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = pl.Trainer(max_epochs=cfg.max_epochs,
                         accelerator="gpu",
                         devices=1,
                         logger=logger,
                         callbacks=[ModelCheckpoint(monitor='val_loss_epoch',
                                                    dirpath=cfg.checkpoint_dir,
                                                    ),
                                    lr_monitor])

    if cfg.train:
        trainer.fit(model, datamodule=datamodule)

    if cfg.test:
        model.eval()
        model = model.to("cpu")
        prompt = "."
        tokenized_prompt = [t.idx for t in datamodule.dataset.source_tokenizer(prompt)]
        tokenized_prompt.insert(0, datamodule.dataset.source_special_characters['<SOS>'])
        tokenized_prompt.append(datamodule.dataset.source_special_characters['<EOS>'])
        tokenized_sentence = torch.LongTensor(tokenized_prompt).unsqueeze(0)

        output = torch.LongTensor([datamodule.dataset.target_special_characters['<SOS>']]).unsqueeze(0)
        end_token = torch.LongTensor([datamodule.dataset.target_special_characters['<EOS>']]).unsqueeze(0)
        print(output.shape, end_token.shape, tokenized_sentence.shape)
        output_sentence  = ['[SOS]']
        while True:
            print("Input sentence", tokenized_sentence)
            print("Start sentence")
            out = model.model(x=tokenized_sentence,
                              y=output)
            out =  Categorical(torch.softmax(out, dim=-1)).sample()
            print(out)
            output[0, 0] = out[0, 0]
            if output[0, 0] == end_token:
                output_sentence.append('[EOS]')
                break
            elif output[0, 0] == datamodule.dataset.target_special_characters['<SOS>']:
                output_sentence.append('[PAD]')
            elif output[0, 0] == datamodule.dataset.target_special_characters['<PAD>']:
                output_sentence.append('[PAD]')
            else:
                word = datamodule.dataset.target_tokenizer.vocab.strings[output[0, 0]]
                output_sentence.append(word)
            print("Output Sentence", output_sentence)


if __name__ == '__main__':
    main()