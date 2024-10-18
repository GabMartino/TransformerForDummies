import json

import torch
import lightning as pl
from torch.utils.data import DataLoader

import linecache as lc
from nltk.tokenize import word_tokenize


def load_vocabulary(vocabulary_path):
    vocabulary = None
    with open(vocabulary_path, "r") as f:
        vocabulary = json.load(f)
    return vocabulary


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
    def __init__(self, source_data_path, source_language, source_vocabulary_path,
                 target_data_path, target_language, target_vocabulary_path,
                 max_sentence_len = 150, max_dataset_lenght=100000):
        super(TextTranslationDatasetOnDemand, self).__init__()

        if source_language not in ["English", "Italian"] and target_language not in ["English", "Italian"]:
            raise NotImplementedError("Only English and Italian are supported for translation")
        self.source_data_path = source_data_path
        self.source_language = source_language
        self.source_vocabulary_path = source_vocabulary_path

        self.target_data_path = target_data_path
        self.target_language = target_language
        self.target_vocabulary_path = target_vocabulary_path

        self.source_vocabulary = load_vocabulary(source_vocabulary_path)
        self.target_vocabulary = load_vocabulary(target_vocabulary_path)

        self.source_vocabulary["[SOS]"] = len(self.source_vocabulary)
        self.source_vocabulary["[EOS]"] = len(self.source_vocabulary)
        self.source_vocabulary["[PAD]"] = len(self.source_vocabulary)

        self.target_vocabulary["[SOS]"] = len(self.target_vocabulary)
        self.target_vocabulary["[EOS]"] = len(self.target_vocabulary)
        self.target_vocabulary["[PAD]"] = len(self.target_vocabulary)

        self.max_dataset_lenght = max_dataset_lenght
        self.max_sentence_len = max_sentence_len + 2
        check_maximum_size(self.source_data_path, max_dataset_lenght)
        check_maximum_size(self.target_data_path, max_dataset_lenght)
        self.source_vocab_size = len(self.source_vocabulary)
        self.target_vocab_size = len(self.target_vocabulary)

    def __len__(self):
        return self.max_dataset_lenght

    def prepare_sentence(self, raw_sentence, vocabulary):
        source_sentence = raw_sentence
        source_sentence.insert(0, vocabulary['[SOS]'])
        source_sentence.append(vocabulary['[EOS]'])
        source_padding = [vocabulary['[PAD]']] * (self.max_sentence_len - len(source_sentence))
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

    def get_next_valid_sentence(self, index, data_path, vocabulary):
        tokenized_sentence = None
        p = 0
        while True:
            raw_sentence = lc.getline(data_path, index + p)
            tokenized_sentence = [vocabulary.get(t.lower(), -1)  for t in word_tokenize(raw_sentence)]
            if len(tokenized_sentence) <= self.max_sentence_len - 2:
                break
            p += 1
            if p >= 50:
                raise ValueError("Not enough sentences with the set size, try to increase max_sentence_lenght")
        return tokenized_sentence


    def __getitem__(self, index):

        raw_source_sentence = self.get_next_valid_sentence(index, self.source_data_path, self.source_vocabulary)
        raw_target_sentence = self.get_next_valid_sentence(index, self.target_data_path, self.target_vocabulary)
        source_sentence, source_padding_mask = self.prepare_sentence(raw_source_sentence, self.source_vocabulary)
        target_sentence, target_padding_mask = self.prepare_sentence(raw_target_sentence, self.target_vocabulary)

        return torch.LongTensor(source_sentence), source_padding_mask, \
                torch.LongTensor(target_sentence), target_padding_mask

class TextTranslationDataloader(pl.LightningDataModule):
    def __init__(self, source_data_path, source_language, source_vocabulary_path,
                 target_data_path, target_language,  target_vocabulary_path,
                 batch_size, max_sentence_len=150, max_dataset_lenght=100000):
        super(TextTranslationDataloader, self).__init__()
        self.batch_size = batch_size
        self.dataset = TextTranslationDatasetOnDemand(source_data_path, source_language, source_vocabulary_path,
                                                      target_data_path, target_language, target_vocabulary_path,
                                                      max_sentence_len, max_dataset_lenght)
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
                                    source_vocabulary_path="english_vocab.json",
                                     target_data_path=italian_dataset_path,
                                     target_language="Italian",
                                             target_vocabulary_path="italian_vocab.json",
                                     max_dataset_lenght=1000)
    for batch in iter(dataset):
        print(batch)
        exit()

if __name__ == '__main__':
    main()