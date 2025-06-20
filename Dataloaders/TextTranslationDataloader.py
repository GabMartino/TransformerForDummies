

import lightning as pl
import torch
from torch.utils.data import DataLoader

from Dataloaders.TextTranslationDataset import TextTranslationDataset


class TextTranslationDataloader(pl.LightningDataModule):
    def __init__(self, source_data_path, source_language, source_vocabulary_path,
                 target_data_path, target_language,  target_vocabulary_path,
                 batch_size,
                 seed: int = 42,
                 split_size: int = 0.8):
        super(TextTranslationDataloader, self).__init__()
        generator = torch.Generator().manual_seed(seed)
        self.batch_size = batch_size
        self.dataset = TextTranslationDataset(source_dataset_path=source_data_path,
                                              source_language=source_language,
                                              source_vocabulary_path=source_vocabulary_path,
                                              target_dataset_path=target_data_path,
                                              target_language=target_language,
                                              target_vocabulary_path=target_vocabulary_path)
        train_size = int(len(self.dataset)*split_size)
        val_size = len(self.dataset) - train_size
        self.train_set, self.val_set = torch.utils.data.random_split(self.dataset, [train_size, val_size], generator=generator)
    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, collate_fn=self.dataset.collate_fn, shuffle=True, prefetch_factor=2, persistent_workers=True, num_workers=4)
    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, collate_fn=self.dataset.collate_fn, shuffle=False, prefetch_factor=2, persistent_workers=True, num_workers=4)


def main():
    pass

if __name__ == '__main__':
    main()