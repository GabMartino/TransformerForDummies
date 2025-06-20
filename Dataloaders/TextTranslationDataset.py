
import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer

'''
    This Dataset consider that every sentence will be align per row ending by \n
    - Create an index to reach quickly the indexed line (Note: use UTF-8 encoding)
'''
class LineCache:
    def __init__(self, filename):
        self.filename = filename
        self.offsets = []
        self.file = open(filename, 'r')  # Keep the file open
        self._build_index()

    def _build_index(self):
        """Build line offsets (byte positions) for fast access."""
        offset = 0
        for idx, line in enumerate(self.file):
            self.offsets.append(offset)
            offset += len(line.encode('utf-8'))
        self.file.seek(0)  # Reset pointer to start

    def get_line(self, line_number):
        if line_number < 0 or line_number >= len(self.offsets):
            return None
        self.file.seek(self.offsets[line_number])
        return self.file.readline().rstrip('\n')

    def __len__(self):
        return len(self.offsets)

    def close(self):
        self.file.close()


class TextTranslationDataset(Dataset):

    def __init__(self, source_dataset_path: str,
                 target_dataset_path: str,
                 source_vocabulary_path: str,
                 target_vocabulary_path: str,
                 source_language: str,
                 target_language: str,):
        super(TextTranslationDataset, self).__init__()
        self.source_dataset_path = source_dataset_path
        self.target_dataset_path = target_dataset_path
        self.source_vocabulary_path = source_vocabulary_path
        self.target_vocabulary_path = target_vocabulary_path

        self.source_lines = LineCache(self.source_dataset_path)
        self.target_lines = LineCache(self.target_dataset_path)
        assert len(self.source_lines) == len(self.target_lines)

        self.source_tokenizer = Tokenizer.from_file(source_vocabulary_path)
        self.target_tokenizer = Tokenizer.from_file(target_vocabulary_path)

    def __len__(self):
        return self.source_lines.__len__()

    def __getitem__(self, idx):

        source_line = self.source_lines.get_line(idx)
        target_line = self.target_lines.get_line(idx)

        source_tokens = self.source_tokenizer.encode(source_line).ids
        target_tokens = self.target_tokenizer.encode(target_line).ids
        return source_tokens, len(source_tokens), target_tokens, len(target_tokens)


    def collate_fn(self, batch):

        source_sequences, source_lenghts,  target_sequences, target_lenghts = zip(*batch)
        max_source_len = max(source_lenghts)
        max_target_len = max(target_lenghts)

        source_sequences_list = []
        source_padding_mask = []
        target_sequences_list = []
        target_padding_mask = []
        for source_sequence, target_sequence in zip(list(source_sequences), list(target_sequences)):
            '''
                Create masks
            '''
            source_mask = [False] * (len(source_sequence) + 2 ) + [True] * (max_source_len - len(source_sequence))
            target_mask = [False] * (len(target_sequence) + 2 )  + [True] * (max_target_len - len(target_sequence))
            source_padding_mask.append(source_mask)
            target_padding_mask.append(target_mask)
            '''
                Add padding
            '''
            source_sequence = self.source_tokenizer.encode("[SOS]").ids + source_sequence + self.source_tokenizer.encode("[EOS]").ids + self.source_tokenizer.encode("[PAD]").ids * (max_source_len - len(source_sequence))
            target_sequence = self.target_tokenizer.encode("[SOS]").ids + target_sequence + self.target_tokenizer.encode("[EOS]").ids + self.target_tokenizer.encode("[PAD]").ids * (max_target_len - len(target_sequence))
            source_sequences_list.append(source_sequence)
            target_sequences_list.append(target_sequence)

        source_sequences = torch.LongTensor(source_sequences_list)
        source_padding_mask = torch.BoolTensor(source_padding_mask)
        target_sequences = torch.LongTensor(target_sequences_list)
        target_padding_mask = torch.BoolTensor(target_padding_mask)
        return source_sequences, source_padding_mask, target_sequences, target_padding_mask




def main():
    source_path = "../Datasets/it-en/europarl-v7.it-en.en"
    target_path = "../Datasets/it-en/europarl-v7.it-en.it"
    source_vocab_path = "bpe_english_vocab.json"
    target_vocab_path = "bpe_italian_vocab.json"
    dataset = TextTranslationDataset(source_path, target_path, source_vocab_path, target_vocab_path, "english", "italian")

    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=2, collate_fn=dataset.collate_fn)
    for batch in dataloader:
        source_sequences, source_padding_mask, target_sequences, target_padding_mask = batch
        print(source_sequences.shape, source_padding_mask.shape, target_sequences.shape, target_padding_mask.shape)
        break


if __name__ == "__main__":
    main()