




import torch
import torch.nn as nn
import spacy
import lightning as pl
from spacy.lang.en.tokenizer_exceptions import verb_data
from spacy.lang.it import Italian
from torch.utils.data import DataLoader
from spacy.tokenizer import Tokenizer
from tqdm import tqdm
import linecache as lc

START_TOKEN = "<START>"
END_TOKEN = "<END>"
PAD_TOKEN = "<PAD>"

def load_dataset(data_path, max_dataset_lenght):
    sentences = []
    with open(data_path, 'r') as f:
        for idx, line in enumerate(f):
            if idx >= max_dataset_lenght:
                break
            sentences.append(line.strip())
    return sentences


def from_padding_vector_to_matrix(padding_vector):
    lenght = padding_vector.shape[0]
    padding_mask = padding_vector.unsqueeze(0).repeat(lenght, 1)
    i, j = torch.triu_indices(lenght, lenght)
    vals = padding_mask[i, j]
    padding_mask = padding_mask.transpose(-2, -1)
    padding_mask[i, j] = vals
    return padding_mask

def check_maximum_size(data_path, maximum_lenght):
    with open(data_path, "rb") as f:
        num_lines = sum(1 for _ in f)
    if num_lines < maximum_lenght:
        raise ValueError("The Maximum lenght size set for the dataset is higher than the \n number of the sentence present.")

class TextTranslationDatasetOnDemand(torch.utils.data.Dataset):
    def __init__(self, source_data_path, source_language, target_data_path, target_language, max_sentence_len = 150, max_dataset_lenght=100000):
        super(TextTranslationDatasetOnDemand, self).__init__()
        self.source_data_path = source_data_path
        self.source_language = source_language
        self.target_data_path = target_data_path
        self.target_language = target_language
        self.max_dataset_lenght = max_dataset_lenght
        self.max_sentence_len = max_sentence_len + 2
        check_maximum_size(self.source_data_path, max_dataset_lenght)
        check_maximum_size(self.target_data_path, max_dataset_lenght)
        print("Loading data...")
        self.supported_languages = {
            "English": spacy.load("en_core_web_md"),
            "Italian": spacy.load("it_core_news_md"),
        }
        if source_language not in ["English", "Italian"] and target_language not in ["English", "Italian"]:
            raise NotImplementedError("Only English and Italian are supported for translation")
        ############### TOKENIZATION
        self.source_tokenizer = self.supported_languages[source_language].tokenizer
        self.target_tokenizer = self.supported_languages[target_language].tokenizer
        self.source_vocab_size = self.source_tokenizer.vocab.vectors.n_keys + 3 ## the START, END and PAD token
        self.target_vocab_size = self.target_tokenizer.vocab.vectors.n_keys + 3
        print(self.source_vocab_size, self.target_vocab_size)
        self.source_special_characters = {
            "<SOS>": self.source_vocab_size - 3,
            "<EOS>": self.source_vocab_size - 2,
            "<PAD>": self.source_vocab_size -1
        }
        self.target_special_characters = {
            "<SOS>": self.target_vocab_size - 3,
            "<EOS>": self.target_vocab_size - 2,
            "<PAD>": self.target_vocab_size - 1
        }
    def __len__(self):
        return self.max_dataset_lenght

    def prepare_source_sentence(self, raw_sentence):
        source_sentence = raw_sentence
        source_sentence.insert(0, self.source_special_characters['<SOS>'])
        source_sentence.append(self.source_special_characters['<EOS>'])
        source_padding = [self.source_special_characters['<PAD>']] * (self.max_sentence_len - len(source_sentence))
        source_sentence += source_padding
        source_padding_vector = torch.full((len(source_sentence),), 0.0)
        source_padding_vector[-len(source_padding):] = -torch.inf
        source_padding_mask = from_padding_vector_to_matrix(source_padding_vector)
        return source_sentence, source_padding_mask

    def prepare_target_sentence(self, raw_sentence):
        target_sentence = raw_sentence
        target_sentence.insert(0, self.target_special_characters['<SOS>'])
        target_sentence.append(self.target_special_characters['<EOS>'])
        target_padding = [self.target_special_characters['<PAD>']] * (self.max_sentence_len - len(target_sentence))
        target_sentence += target_padding
        target_padding_vector = torch.full((len(target_sentence),), 0.0)
        target_padding_vector[-len(target_padding):] = -torch.inf
        target_padding_mask = from_padding_vector_to_matrix(target_padding_vector)
        return target_sentence, target_padding_mask

    def get_next_valid_source_sentence(self, index):
        source_tokenized_sentence = None
        p = 0
        while True:
            source_raw_sentence = lc.getline(self.source_data_path, index + p)
            source_tokenized_sentence = [t.idx for t in self.source_tokenizer(source_raw_sentence)]
            if len(source_tokenized_sentence) <= self.max_sentence_len - 2:
                break
            p += 1
            if p >= 50:
                raise ValueError("Not enough sentences with the set size, try to increase max_sentence_lenght")
        return source_tokenized_sentence
    def get_next_valid_target_sentence(self, index):
        target_tokenized_sentence = None
        p = 0
        while True:
            target_raw_sentence = lc.getline(self.target_data_path, index + p)
            target_tokenized_sentence = [t.idx for t in self.source_tokenizer(target_raw_sentence)]
            if len(target_tokenized_sentence) <= self.max_sentence_len - 2:
                break
            p += 1
            if p >= 50:
                raise ValueError("Not enough sentences with the set size, try to increase max_sentence_lenght")
        return target_tokenized_sentence
    def __getitem__(self, index):
        '''

            Notice:
            each source-target couple NOT ONLY will have the same size
            but all the sentences in a batch will have the same size.
            This because the sentences in a batch should have the same size to be stack
        '''

        raw_source_sentence = self.get_next_valid_source_sentence(index)
        raw_target_sentence = self.get_next_valid_target_sentence(index)
        source_sentence, source_padding_mask = self.prepare_source_sentence(raw_source_sentence)
        target_sentence, target_padding_mask = self.prepare_target_sentence(raw_target_sentence)

        return torch.LongTensor(source_sentence), source_padding_mask, \
                torch.LongTensor(target_sentence), target_padding_mask

class TextTranslationDataloader(pl.LightningDataModule):
    def __init__(self, source_data_path, source_language, target_data_path, target_language, batch_size, max_sentence_len=150, max_dataset_lenght=100000):
        super(TextTranslationDataloader, self).__init__()
        self.batch_size = batch_size
        self.dataset = TextTranslationDatasetOnDemand(source_data_path, source_language, target_data_path, target_language, max_sentence_len, max_dataset_lenght)
        train_size = int(len(self.dataset)*0.8)
        val_size = len(self.dataset) - train_size
        self.train_set, self.val_set = torch.utils.data.random_split(self.dataset, [train_size, val_size])
    def train_dataloader(self):
        def collate(batch):
            source_batches = []
            source_masks = []
            target_batches = []
            target_masks = []
            for element in batch:
                source_batch, source_mask, target_batch, target_mask = element
                #print(source_batch.shape, source_mask.shape, target_batch.shape, target_mask.shape)
                source_batches.append(source_batch)
                source_masks.append(source_mask)
                target_batches.append(target_batch)
                target_masks.append(target_mask)

            return (torch.stack(source_batches),
                    torch.stack(source_masks),
                    torch.stack(target_batches),
                    torch.stack(target_masks))
        return DataLoader(self.train_set, batch_size=self.batch_size, collate_fn=collate, shuffle=True, num_workers=4)
    def val_dataloader(self):
        def collate(batch):
            source_batches = []
            source_masks = []
            target_batches = []
            target_masks = []
            for element in batch:
                source_batch, source_mask, target_batch, target_mask = element
                source_batches.append(source_batch)
                source_masks.append(source_mask)
                target_batches.append(target_batch)
                target_masks.append(target_mask)
            return (torch.stack(source_batches),
                    torch.stack(source_masks),
                    torch.stack(target_batches),
                    torch.stack(target_masks))
        return DataLoader(self.val_set, batch_size=self.batch_size, collate_fn=collate, shuffle=False, num_workers=4)


def main():
    english_dataset_path = "../Datasets/it-en/europarl-v7.it-en.en"
    italian_dataset_path = "../Datasets/it-en/europarl-v7.it-en.it"

    dataset = TextTranslationDatasetOnDemand(source_data_path=english_dataset_path,
                                     source_language="English",
                                     target_data_path=italian_dataset_path,
                                     target_language="Italian",
                                     max_dataset_lenght=1000)
    for batch in iter(dataset):
        break
    dataloader = TextTranslationDataloader(source_data_path=english_dataset_path,
                                           source_language="English",
                                           target_data_path=italian_dataset_path,
                                           target_language="Italian",
                                           batch_size=3,
                                           max_sentence_len= 150,
                                           max_dataset_lenght=10000)
    for batch in iter(dataloader.train_dataloader()):
        print(batch[0].shape, batch[1].shape, batch[2].shape, batch[3].shape)
        break
if __name__ == '__main__':
    main()