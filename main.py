import glob
import os

import hydra
import lightning as pl
import torch
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from nltk import word_tokenize

from Dataloaders.TextTranslationDataloader import TextTranslationDataloader
from model.TransformerLightning import TransformerLightning
from model.utils.utils import token_to_text


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
                                           source_vocabulary_path=cfg.source_language.vocabulary_path,
                                           target_data_path=cfg.target_language.dataset_path,
                                           target_language=cfg.target_language.name,
                                           target_vocabulary_path=cfg.target_language.vocabulary_path,
                                           batch_size=cfg.batch_size,
                                           max_dataset_lenght=cfg.maximum_dataset_lenght)


    '''
        These variable are necessary to implement the model (embedding size in the embedding layers)
    '''
    cfg.encoder.vocab_size = datamodule.dataset.source_vocab_size
    cfg.decoder.vocab_size = datamodule.dataset.target_vocab_size
    '''
        This variable is necessary to set the ignore index in the cross entropy loss 
    '''
    cfg.ignore_index = datamodule.dataset.target_vocabulary["[PAD]"]


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
        model = model.model
        print("model size", model.target_embedding.linear_out.out_features)
        prompt = "bridge"
        '''
            1. Tokenize
        '''
        vocabulary = datamodule.dataset.source_vocabulary
        tokenized_sentence = [vocabulary[t.lower()] for t in word_tokenize(prompt)]
        '''
            2. Let's insert the [SOS] and [EOS] token in the source sentence
        '''
        tokenized_sentence.insert(0, vocabulary["[SOS]"])
        tokenized_sentence.append(vocabulary["[EOS]"])
        '''
            3. Let's create the input for the decoder 
        '''
        decoder_input = [datamodule.dataset.target_vocabulary["[SOS]"]]
        print(tokenized_sentence, decoder_input)
        print(token_to_text(tokenized_sentence, vocabulary), token_to_text(decoder_input, datamodule.dataset.target_vocabulary))

        tokenized_sentence = torch.LongTensor(tokenized_sentence, device="cpu").unsqueeze(0)
        decoder_input = torch.LongTensor(decoder_input, device="cpu").unsqueeze(0)
        while True:
            out = model(x=tokenized_sentence, y=decoder_input)
            out = torch.argmax(torch.softmax(out, dim=-1)) ## I'm just using the argmax, i'm not sampling
            decoder_input = torch.cat([decoder_input, out.unsqueeze(0).unsqueeze(0)], dim=-1)
            if out == datamodule.dataset.target_vocabulary["[EOS]"]:
                break
            print(token_to_text(decoder_input.squeeze(0).numpy(), datamodule.dataset.target_vocabulary))





if __name__ == '__main__':
    main()